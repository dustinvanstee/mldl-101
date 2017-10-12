import timeit

def time_run(command_str) :
    #rv=%timeit -o -n 1 command
    rv= timeit.Timer(stmt=command_str).timeit(number=1)
    return rv


def buildGpuCpuMmStrings(dim1,dim2,dim3) :
    '''
    def buildGpuCpuMmStrings(dim1,dim2,dim3)
        Builds a python command string for use with timeit
        
        String multiplys two matrices of dimensions [dim1 x dim2]  * [dim2 x dim3] 
        and returns the runtimes for GPU vs CPU
    
    Inputs 
      - dim1 : matrix1 rows
      - dim2 : matrix1 cols, matrix2 rows
      - dim3 : matrix 2 cols

    Returns
      - tuple of Strings (GPURUN:string,CPURUN:string)
    '''
    
    # Convert int to string 
    dim1 = str(dim1)
    dim2 = str(dim2)
    dim3 = str(dim3)

    #  The timeit function 
    GPURUN = """
import tensorflow as tf
tf.reset_default_graph() 

mat1 = tf.random_normal([""" + dim1 + ',' +dim2 + """],mean=0.0, stddev=1.0, dtype=tf.float32)
mat2 = tf.random_normal([""" + dim2 + ',' +dim3 + """],mean=0.0, stddev=1.0, dtype=tf.float32)
mat3 = tf.matmul(mat1, mat2)

# Creates a session with log_device_placement set to True.
with tf.Session() as sess :
    writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    sess.run(mat3)
    sess.close()
    """
    
    CPURUN = """
import tensorflow as tf
tf.reset_default_graph() 

with tf.device('/cpu:0'):
    mat1 = tf.random_normal([""" + dim1 + ',' +dim2 + """],mean=0.0, stddev=1.0, dtype=tf.float32)
    mat2 = tf.random_normal([""" + dim2 + ',' +dim3 + """],mean=0.0, stddev=1.0, dtype=tf.float32)
    mat3 = tf.matmul(mat1, mat2)

    # Creates a session with log_device_placement set to True.
with tf.Session() as sess :
    writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    sess.run(mat3)
    sess.close()
    """
    return (GPURUN,CPURUN)

def buildGpuCpuStringsForTimeRun(n) :
    '''
    def buildGpuCpuStringsForTimeRun(n)
        Multiplys two square matrices of [n x n]
    Inputs 
      - n (int) , defines matrix1 and matrix2 square dimensions

    Returns
      - tuple of strings (GPURUN:string,CPURUN:string)
    '''
    
    return buildGpuCpuMmStrings(n,n,n)

def getRatio(n) :
    if(n > 25000) :
        print "Size > 25000 will take to long to run.  Returning 0 for this run"
        return 0
    else :
        (GPURUN,CPURUN) = buildGpuCpuStringsForTimeRun(n)
        gpu_time=time_run(GPURUN)
        cpu_time=time_run(CPURUN)
        speedup = cpu_time / gpu_time
        return speedup
