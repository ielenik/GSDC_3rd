import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import dtype

tf.keras.backend.set_floatx('float64')

K1 = 0.01
K2 = 0.01
K3 = 0.01

def set_bias_regulizer(model):
    def kernel_init(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[:,0,0] = np.array([-1,1]).astype(np.float64)
        return kernel
    
    derivative = tf.keras.layers.Conv1D(1,2,use_bias=False,kernel_initializer=kernel_init, dtype = tf.float64)
    derivative.trainable = False

    def custom_bias_regularizer(weights):
        wt = tf.transpose(weights)
        wt = tf.reshape(wt,[1,-1,1])
        speed = derivative(wt)
        return 0.00001*tf.reduce_sum(tf.square(speed))

    model.add_loss(lambda layer=model.get_layer("time_bias"): custom_bias_regularizer(layer.kernel))            


def createModel(numepochs, numsatelite, numseg, in_pos, umbig_init):
    epoch        = tf.keras.layers.Input((1), dtype=tf.int32)
    satind       = tf.keras.layers.Input((1), dtype=tf.int32)
    segind       = tf.keras.layers.Input((1), dtype=tf.int32)
    sat_pos      = tf.keras.layers.Input((3))
    
    initpos = np.array(in_pos).astype(np.float64)
    
  
    def edge_constrain(weights):
        weights[0,:].assign(weights[1,:])
        weights[-1,:].assign(weights[-2,:])
        return weights

    positions = tf.keras.layers.Dense(3,use_bias=False, kernel_initializer=tf.keras.initializers.Constant(initpos), kernel_constraint=edge_constrain, name = 'positions', dtype = tf.float64)
    umbiguites = tf.keras.layers.Dense(1,use_bias=False, kernel_initializer=tf.keras.initializers.Constant(umbig_init), name = 'umbiguites')#, kernel_regularizer=tf.keras.regularizers.L2(0.00001))
    time_bias = tf.keras.layers.Dense(1,use_bias=False, kernel_initializer=tf.keras.initializers.Zeros(), name = 'time_bias')#, kernel_regularizer=tf.keras.regularizers.L2(0.00001))
    track_shift = tf.Variable([[0.,0.,0.]],trainable=True,name="track_shift",dtype=tf.float64)

    def predict_epoch(epoch_z0, pos_in, numepochs):
        epoch_z0 = tf.one_hot(epoch_z0, numepochs, dtype = tf.float64)
        epoch_z0 = tf.squeeze(epoch_z0, axis = 1)
        epoch_z0 = pos_in(epoch_z0)
        return epoch_z0

    epoch_z0 = predict_epoch(epoch, positions, numepochs)
    epoch_m1 = predict_epoch(epoch - 1, positions, numepochs)
    epoch_p1 = predict_epoch(epoch + 1, positions, numepochs)
    

    bias  = predict_epoch(epoch, time_bias, numepochs)
    umbig = predict_epoch(segind, umbiguites, numseg)

    shift = tf.linalg.norm(epoch_z0+track_shift - sat_pos , axis = -1, keepdims=True) 
    shift = shift + bias + umbig


    model = tf.keras.Model([epoch,satind,segind, sat_pos], shift)
    #set_bias_regulizer(model)
    acsel_los = tf.reduce_mean((epoch_m1+epoch_p1-2*epoch_z0)**2)/10
    #model.add_loss(acsel_los)
    model.add_metric(acsel_los, name = "acs")
    #model.add_metric(tf.reduce_mean(weights), name = "acs1")
    #model.add_loss(tf.reduce_mean(tf.nn.relu(tf.abs(tf.norm(in_pos[0], axis = -1) - tf.norm(epoch_z0, axis = -1))-40))*0.1)

    return model, tf.keras.Model(epoch,epoch_z0+track_shift), acsel_los

def set_position_regulizer(model):
    def kernel_init(shape, dtype=None, partition_info=None):
        kernel = np.zeros(shape)
        kernel[:,0,0] = np.array([-1,1]).astype(np.float64)
        return kernel
    
    derivative = tf.keras.layers.Conv1D(1,2,use_bias=False,kernel_initializer=kernel_init, dtype = tf.float64)
    derivative.trainable = False

    def custom_accel_regularizer(weights):
        wt = tf.transpose(weights)
        wt = tf.reshape(wt,[3,-1,1])
        speed = derivative(wt)
        accel = derivative(speed)
        d3 = derivative(accel)
        return 0.01*(tf.reduce_sum(tf.abs(accel)) + tf.reduce_sum(tf.abs(d3))) + tf.reduce_sum(tf.nn.relu(tf.abs(accel)-15)) + tf.reduce_sum(tf.nn.relu(tf.abs(speed)-40))

    l = model.get_layer("positions")
    model.add_loss(lambda layer=model.get_layer("positions"): custom_accel_regularizer(layer.kernel))            
    #model.add_metric(custom_accel_regularizer(l.kernel), "accel")            
