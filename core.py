import numpy as np
from scipy.signal import lfilter
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfpd
from tensorflow_probability import bijectors as tfpb

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, use_bias=True):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation, use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer())

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def get_sddpg_actor_critic(ac_type):
    if ac_type == "mlp":
        return sddpg_actor_critic
    else:
        raise Exception("wrong ac_type")

def get_ddpg_actor_critic(ac_type):
    if ac_type == "mlp":
        return ddpg_actor_critic
    else:
        raise Exception("wrong ac_type")

def get_td3_actor_critic(ac_type):
    if ac_type == "mlp":
        return td3_actor_critic
    else:
        raise Exception("wrong ac_type")

def get_iac_actor_critic(ac_type):
    if ac_type == "mlp":
        return iac_actor_critic
    else:
        raise Exception("wrong ac_type")

def get_ppo_actor_critic(ac_type):
    if ac_type == "gaussian":
        return gaussian_actor_critic
    elif ac_type == "gaussian_mixture":
        return gaussian_mixture_actor_critic
    elif ac_type == "affine_flow":
        return affine_flow_actor_critic
    else:
        raise Exception("wrong ac_type")

def ddpg_actor_critic(x, a, pi_s, layers=[20, 20]):
    act_dim = a.shape.as_list()[-1]
    
    with tf.variable_scope('pi'):
        if pi_s:
            pi = mlp(x, layers + [act_dim], tf.nn.leaky_relu, tf.nn.sigmoid)
        else:
            pi = mlp(tf.ones_like(x), layers + [act_dim], tf.nn.leaky_relu, tf.nn.sigmoid, use_bias=False)
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), layers + [1], tf.nn.leaky_relu, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), layers + [1], tf.nn.leaky_relu, None), axis=1)
    return pi, q, q_pi

def sddpg_actor_critic(s, a, z, pi_sz, layers=[20, 20]):
    act_dim = a.shape.as_list()[-1]

    with tf.variable_scope('pi'):
        if pi_sz:
            pi = mlp(tf.concat([s, z], axis=-1), layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid)
        else:
            pi = mlp(z, layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid, use_bias=False)

    vf_mlp = lambda x: tf.squeeze(mlp(x, layers+[1], tf.nn.leaky_relu, None), axis=1)
    with tf.variable_scope('q'):
        q = vf_mlp(tf.concat([s, a], axis=-1))
    with tf.variable_scope('q', reuse=True):
        q_pi = vf_mlp(tf.concat([s, pi], axis=-1))
    with tf.variable_scope('v'):
        v = vf_mlp(s)
    return pi, q, q_pi, v

def td3_actor_critic(s, a, z, pi_sz, layers=[20, 20]):
    act_dim = a.shape.as_list()[-1]
    vf_mlp = lambda x: tf.squeeze(mlp(x, layers+[1], tf.nn.leaky_relu, None), axis=1)
    with tf.variable_scope('pi'):
        if pi_sz:
            pi = mlp(tf.concat([s, z], axis=-1), layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid)
        else:
            pi = mlp(z, layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid, use_bias=False)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([s, a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([s, pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([s, a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([s, pi], axis=-1))
    q = tf.minimum(q1, q2)
    with tf.variable_scope('v1'):
        v1 = vf_mlp(s)
    with tf.variable_scope('v2'):
        v2 = vf_mlp(s)
    v = tf.minimum(v1, v2)
    return pi, q1, q2, q, q1_pi, q2_pi, v1, v2, v,


def iac_actor_critic(s, a, z, pi_sz, pi_separate, use_bias, layers=[20, 20]):
    act_dim = a.shape.as_list()[-1]
    z_dim = z.shape.as_list()[-1]
    vf_mlp = lambda x: tf.squeeze(mlp(x, layers+[1], tf.nn.leaky_relu, None), axis=1)
    with tf.variable_scope('pi'):
        if pi_sz:
            if pi_separate:
                with tf.variable_scope('emb'):
                    emb = mlp(s, layers+[z_dim], tf.nn.leaky_relu, tf.nn.leaky_relu)
                with tf.variable_scope('d'):
                    pi = mlp(tf.concat([emb, z], axis=-1), layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid, use_bias=use_bias)
                    #pi = mlp(emb*z, layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid)
            else:
                pi = mlp(tf.concat([s, z], axis=-1), layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid)
                #pi = mlp(s*z, layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid)
        else:
            pi = mlp(z, layers+[act_dim], tf.nn.leaky_relu, tf.nn.sigmoid, use_bias=use_bias)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([s, a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([s, pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([s, a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([s, pi], axis=-1))
    with tf.variable_scope('v'):
        v = vf_mlp(s)
    return pi, q1, q2, q1_pi, q2_pi, v

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def gaussian_mixture_likelihood(x, k, act_dim, p, mu, std):
    mu = tf.tile(mu, [tf.shape(x)[0], 1])
    std = tf.tile(std, [tf.shape(x)[0], 1])
    p = tf.tile(p, [tf.shape(x)[0]])
    x = tf.reshape(tf.tile(x, [1, k]), [-1, act_dim])
    probs = tf.reciprocal((std+EPS)*np.sqrt(2*np.pi)) * tf.exp(-0.5*((x-mu)/(std+EPS))**2)
    probs = tf.reduce_prod(probs, axis=1) * tf.exp(p)
    return tf.log(tf.reduce_sum(tf.reshape(probs, [-1, k]), axis=1))

def cum_weighted_sum(x, weight):
    return lfilter(weight, [1.], x[::-1])[::-1]

def gaussian_actor_critic(s, a):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(s, [act_dim], None, None, use_bias=False)
    log_std = tf.get_variable(name="log_std", initializer=-2*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

def gaussian_mixture_actor_critic(a, k):
    act_dim = a.shape.as_list()[-1]
    p = tf.get_variable(name="p", initializer=np.ones(k, dtype=np.float32) / k)
    mu = tf.get_variable(name="mu", shape=(k, act_dim), initializer=tf.truncated_normal_initializer(0.5, 0.2))
    #mu = tf.get_variable(name="mu", initializer=np.array([[0.433,0.433],[0.6,0.6]], dtype=np.float32))
    log_std = tf.get_variable(name="log_std", initializer=-2*np.ones((k, act_dim), dtype=np.float32))
    std = tf.exp(log_std)

    select = tf.squeeze(tf.multinomial(tf.expand_dims(p, 0), 1))
    pi = tf.gather(mu, select) + tf.random_normal((act_dim,)) * tf.gather(std, select)
    pi = tf.expand_dims(pi, 0)

    logp = gaussian_mixture_likelihood(a, k, act_dim, p, mu, std)
    logp_pi = gaussian_mixture_likelihood(pi, k, act_dim, p, mu, std)
    return pi, logp, logp_pi

class PReLU(tfpb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="prelu"):
        super(PReLU, self).__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        ones = tf.ones_like(y)
        return tf.log(tf.where(tf.greater_equal(y, 0), ones, ones / self.alpha))

def affine_flow_actor_critic(a, k):
    d = r = act_dim = a.shape.as_list()[-1]
    DTYPE = tf.float32
    bijectors = []
    initializer = tf.initializers.truncated_normal(0, 0.1)
    for i in range(k):
        with tf.variable_scope('bijector_%d' % i):
            V = tf.get_variable('V', [d, r], dtype=DTYPE, initializer=initializer)
            shift = tf.get_variable('shift', [d], dtype=DTYPE, initializer=initializer)
            L = tf.get_variable('L', [d * (d + 1) / 2], dtype=DTYPE, initializer=initializer)
            bijectors.append(tfpb.Affine(
                scale_tril=tfpd.fill_triangular(L),
                scale_perturb_factor=V,
                shift=shift,
            ))
            alpha = tf.abs(tf.get_variable('alpha', [], dtype=DTYPE)) + .01
            bijectors.append(PReLU(alpha=alpha))
    mlp_bijector = tfpb.Chain(
        list(reversed(bijectors[:-1])), name='mlp_bijector')
    dist = tfpd.TransformedDistribution(
        distribution=tfpd.MultivariateNormalDiag(loc=tf.zeros(act_dim), scale_diag=0.1*tf.ones(act_dim)),
        bijector=mlp_bijector
    )
    pi = dist.sample(1)
    logp_pi = tf.squeeze(dist.log_prob(pi))
    logp = dist.log_prob(a)
    return pi, logp, logp_pi