TF_INC :=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB :=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
INC=-I${TF_INC}
LIB=-L${TF_LIB}
non_max_suppression_op_gpu_op: register_custom_ops.cc non_max_suppression_custom_op_gpu.cu.cc non_max_suppression_custom_op.h non_max_suppression_custom_op.cc
	nvcc -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG -c -o non_max_suppression_custom_op_gpu.cu.o non_max_suppression_custom_op_gpu.cu.cc \
	$(INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -gencode=arch=compute_35,code=sm_35 \
	-gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 \
    	-gencode=arch=compute_70,code=sm_70
	g++ -g -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o non_max_suppression_custom_op.so register_custom_ops.cc \
        non_max_suppression_custom_op.h non_max_suppression_custom_op.cc non_max_suppression_custom_op_gpu.cu.o \
	$(INC) $(LIB) -ltensorflow_framework -L /usr/local/cuda/lib64/ -fPIC -lcudart -O2

clean: *.so
	rm *.so *.o
