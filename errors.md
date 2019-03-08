# Error

### 8 GPU, BS=8, FP16, 1024 loss scaling

No error same config without FP16

```
2019-03-08 11:43:52.915677: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcublas.so.10.0 locally
2019-03-08 11:43:53.016341: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcublas.so.10.0 locally
2019-03-08 11:43:53.078111: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcublas.so.10.0 locally
2019-03-08 11:43:53.103683: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcublas.so.10.0 locally
2019-03-08 11:43:53.117890: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcublas.so.10.0 locally
2019-03-08 11:43:53.174997: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcudnn.so.7 locally
2019-03-08 11:43:53.295756: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcudnn.so.7 locally
2019-03-08 11:43:53.364771: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcublas.so.10.0 locally
2019-03-08 11:43:53.413652: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcudnn.so.7 locally
2019-03-08 11:43:53.478467: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcudnn.so.7 locally
2019-03-08 11:43:53.501384: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcudnn.so.7 locally
2019-03-08 11:43:53.576262: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcudnn.so.7 locally
2019-03-08 11:43:53.756500: I tensorflow/stream_executor/platform/default/dso_loader.cc:161] successfully opened CUDA library libcudnn.so.7 locally
2019-03-08 11:43:55.542161: E tensorflow/stream_executor/cuda/cuda_blas.cc:694] failed to run cuBLAS routine cublasSgemmEx: CUBLAS_STATUS_EXECUTION_FAILED
2019-03-08 11:43:55.543024: I tensorflow/stream_executor/stream.cc:4787] [stream=0x3e5aaf60,impl=0x3e5ab000] did not memcpy device-to-host; source: 0x7ff727723000
2019-03-08 11:43:55.543839: I tensorflow/stream_executor/stream.cc:4825] [stream=0x3e5aaf60,impl=0x3e5ab000] did not memzero GPU location; source: 0x7ff84dffdc00
2019-03-08 11:43:55.543868: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7ff84dffdbf0
2019-03-08 11:43:55.543879: I tensorflow/stream_executor/stream.cc:1826] [stream=0x3e5aaf60,impl=0x3e5ab000] did not enqueue 'start timer': 0x7ff84dffdbf0
2019-03-08 11:43:55.543901: I tensorflow/stream_executor/stream.cc:1838] [stream=0x3e5aaf60,impl=0x3e5ab000] did not enqueue 'stop timer': 0x7ff84dffdbf0
2019-03-08 11:43:55.543911: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
[f1f0059c329e:60771] *** Process received signal ***
[f1f0059c329e:60771] Signal: Aborted (6)
[f1f0059c329e:60771] Signal code:  (-6)
[f1f0059c329e:60771] [ 0] /lib/x86_64-linux-gnu/libpthread.so.0(+0x11390)[0x7ff988c66390]
[f1f0059c329e:60771] [ 1] /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x38)[0x7ff9881b0428]
[f1f0059c329e:60771] [ 2] /lib/x86_64-linux-gnu/libc.so.6(abort+0x16a)[0x7ff9881b202a]
[f1f0059c329e:60771] [ 3] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(+0x6c5ce04)[0x7ff8d36cae04]
[f1f0059c329e:60771] [ 4] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer22GetElapsedMillisecondsEv+0x97)[0x7ff8cc4de507]
[f1f0059c329e:60771] [ 5] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer12MicrosecondsEv+0x9)[0x7ff8cc447959]
[f1f0059c329e:60771] [ 6] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(_ZN10tensorflow10BiasGradOpIN5Eigen9GpuDeviceENS1_4halfEE7ComputeEPNS_15OpKernelContextE+0x33a)[0x7ff8d05dad5a]
[f1f0059c329e:60771] [ 7] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice13ComputeHelperEPNS_8OpKernelEPNS_15OpKernelContextE+0x48a)[0x7ff8cbfb9a6a]
[f1f0059c329e:60771] [ 8] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice7ComputeEPNS_8OpKernelEPNS_15OpKernelContextE+0x2a)[0x7ff8cbfba78a]
[f1f0059c329e:60771] [ 9] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dbe0)[0x7ff8cc010be0]
[f1f0059c329e:60771] [10] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dc6f)[0x7ff8cc010c6f]
[f1f0059c329e:60771] [11] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN5Eigen15ThreadPoolTemplIN10tensorflow6thread16EigenEnvironmentEE10WorkerLoopEi+0x2e2)[0x7ff8cc09fc72]
[f1f0059c329e:60771] [12] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNSt17_Function_handlerIFvvEZN10tensorflow6thread16EigenEnvironment12CreateThreadESt8functionIS0_EEUlvE_E9_M_invokeERKSt9_Any_data+0x48)[0x7ff8cc09ce68]
[f1f0059c329e:60771] [13] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xb8c80)[0x7ff8df6b3c80]
[f1f0059c329e:60771] [14] /lib/x86_64-linux-gnu/libpthread.so.0(+0x76ba)[0x7ff988c5c6ba]
[f1f0059c329e:60771] [15] /lib/x86_64-linux-gnu/libc.so.6(clone+0x6d)[0x7ff98828241d]
[f1f0059c329e:60771] *** End of error message ***
2019-03-08 11:43:55.782540: E tensorflow/stream_executor/cuda/cuda_blas.cc:694] failed to run cuBLAS routine cublasSgemmEx: CUBLAS_STATUS_EXECUTION_FAILED
2019-03-08 11:43:55.782660: I tensorflow/stream_executor/stream.cc:1826] [stream=0x2706ab70,impl=0x2706ac10] did not enqueue 'start timer': 0x7fb77b7f8bf0
2019-03-08 11:43:55.782786: I tensorflow/stream_executor/stream.cc:1838] [stream=0x2706ab70,impl=0x2706ac10] did not enqueue 'stop timer': 0x7fb77b7f8bf0
2019-03-08 11:43:55.783479: I tensorflow/stream_executor/stream.cc:4825] [stream=0x2706ab70,impl=0x2706ac10] did not memzero GPU location; source: 0x7fb77b7f8c00
2019-03-08 11:43:55.783501: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7fb77b7f8bf0
2019-03-08 11:43:55.783509: I tensorflow/stream_executor/stream.cc:1826] [stream=0x2706ab70,impl=0x2706ac10] did not enqueue 'start timer': 0x7fb77b7f8bf0
2019-03-08 11:43:55.783525: I tensorflow/stream_executor/stream.cc:1838] [stream=0x2706ab70,impl=0x2706ac10] did not enqueue 'stop timer': 0x7fb77b7f8bf0
2019-03-08 11:43:55.783533: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
[f1f0059c329e:60772] *** Process received signal ***
[f1f0059c329e:60772] Signal: Aborted (6)
[f1f0059c329e:60772] Signal code:  (-6)
[f1f0059c329e:60772] [ 0] /lib/x86_64-linux-gnu/libpthread.so.0(+0x11390)[0x7fb8b5c57390]
[f1f0059c329e:60772] [ 1] /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x38)[0x7fb8b51a1428]
[f1f0059c329e:60772] [ 2] /lib/x86_64-linux-gnu/libc.so.6(abort+0x16a)[0x7fb8b51a302a]
[f1f0059c329e:60772] [ 3] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(+0x6c5ce04)[0x7fb8006bbe04]
[f1f0059c329e:60772] [ 4] 2019-03-08 11:43:55.785335: I tensorflow/stream_executor/stream.cc:4825] [stream=0x2706ab70,impl=0x2706ac10] did not memzero GPU location; source: 0x7fb77aff7c00
2019-03-08 11:43:55.785363: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7fb77aff7bf0
2019-03-08 11:43:55.785371: I tensorflow/stream_executor/stream.cc:1826] [stream=0x2706ab70,impl=0x2706ac10] did not enqueue 'start timer': 0x7fb77aff7bf0
2019-03-08 11:43:55.785387: I tensorflow/stream_executor/stream.cc:1838] [stream=0x2706ab70,impl=0x2706ac10] did not enqueue 'stop timer': 0x7fb77aff7bf0
2019-03-08 11:43:55.785395: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
/usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer22GetElapsedMillisecondsEv+0x97)[0x7fb7f94cf507]
[f1f0059c329e:60772] [ 5] 2019-03-08 11:43:55.822062: E tensorflow/stream_executor/cuda/cuda_blas.cc:694] failed to run cuBLAS routine cublasSgemmEx: CUBLAS_STATUS_EXECUTION_FAILED
2019-03-08 11:43:55.822129: I tensorflow/stream_executor/stream.cc:1852] [stream=0x25e49360,impl=0x25e49400] did not wait for [stream=0x3e210f80,impl=0x3e211020]
2019-03-08 11:43:55.822179: F tensorflow/core/common_runtime/gpu/gpu_util.cc:292] GPU->CPU Memcpy failed
[f1f0059c329e:60775] *** Process received signal ***
[f1f0059c329e:60775] Signal: Aborted (6)
[f1f0059c329e:60775] Signal code:  (-6)
2019-03-08 11:43:55.822318: I tensorflow/stream_executor/stream.cc:4787] [stream=0x3e210f80,impl=0x3e211020] did not memcpy device-to-host; source: 0x7fe734676500
2019-03-08 11:43:55.822369: I tensorflow/stream_executor/stream.cc:4787] [stream=0x3e210f80,impl=0x3e211020] did not memcpy device-to-host; source: 0x7fe734676500
[f1f0059c329e:60775] [ 0] /lib/x86_64-linux-gnu/libpthread.so.0(+0x11390)[0x7fe99872b390]
[f1f0059c329e:60775] [ 1] /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x38)[0x7fe997c75428]
[f1f0059c329e:60775] [ 2] /lib/x86_64-linux-gnu/libc.so.6(abort+0x16a)[0x7fe997c7702a]
[f1f0059c329e:60775] [ 3] 2019-03-08 11:43:55.822759: I tensorflow/stream_executor/stream.cc:4825] [stream=0x3e210f80,impl=0x3e211020] did not memzero GPU location; source: 0x7fe85d7fcc00
2019-03-08 11:43:55.822774: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7fe85d7fcbf0
2019-03-08 11:43:55.822783: I tensorflow/stream_executor/stream.cc:1826] [stream=0x3e210f80,impl=0x3e211020] did not enqueue 'start timer': 0x7fe85d7fcbf0
2019-03-08 11:43:55.822801: I tensorflow/stream_executor/stream.cc:1838] [stream=0x3e210f80,impl=0x3e211020] did not enqueue 'stop timer': 0x7fe85d7fcbf0
2019-03-08 11:43:55.822808: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
2019-03-08 11:43:55.885675: E tensorflow/stream_executor/cuda/cuda_blas.cc:694] failed to run cuBLAS routine cublasSgemmEx: CUBLAS_STATUS_EXECUTION_FAILED
2019-03-08 11:43:55.887784: I tensorflow/stream_executor/stream.cc:4825] [stream=0x27994770,impl=0x27994810] did not memzero GPU location; source: 0x7f48cc7fac00
2019-03-08 11:43:55.887813: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7f48cc7fabf0
2019-03-08 11:43:55.887821: I tensorflow/stream_executor/stream.cc:1826] [stream=0x27994770,impl=0x27994810] did not enqueue 'start timer': 0x7f48cc7fabf0
2019-03-08 11:43:55.887841: I tensorflow/stream_executor/stream.cc:1838] [stream=0x27994770,impl=0x27994810] did not enqueue 'stop timer': 0x7f48cc7fabf0
2019-03-08 11:43:55.887849: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
[f1f0059c329e:60773] *** Process received signal ***
[f1f0059c329e:60773] Signal: Aborted (6)
[f1f0059c329e:60773] Signal code:  (-6)
2019-03-08 11:43:55.887875: I tensorflow/stream_executor/stream.cc:4787] [stream=0x27994770,impl=0x27994810] did not memcpy device-to-host; source: 0x7f47ab0c7b00
2019-03-08 11:43:55.887924: I tensorflow/stream_executor/stream.cc:4787] [stream=0x27994770,impl=0x27994810] did not memcpy device-to-host; source: 0x7f47ab0c7b00
2019-03-08 11:43:55.887963: I tensorflow/stream_executor/stream.cc:4787] [stream=0x27994770,impl=0x27994810] did not memcpy device-to-host; source: 0x7f47ab0c7b00
[f1f0059c329e:60773] [ 0] /lib/x86_64-linux-gnu/libpthread.so.0(+0x11390)[0x7f4a07216390]
[f1f0059c329e:60773] [ 1] /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x38)[0x7f4a06760428]
[f1f0059c329e:60773] [ 2] /lib/x86_64-linux-gnu/libc.so.6(abort+0x16a)[0x7f4a0676202a]
[f1f0059c329e:60773] [ 3] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(+0x6c5ce04)[0x7f4951c64e04]
[f1f0059c329e:60773] [ 4] 2019-03-08 11:43:55.889724: E tensorflow/stream_executor/cuda/cuda_blas.cc:694] failed to run cuBLAS routine cublasSgemmEx: CUBLAS_STATUS_EXECUTION_FAILED
2019-03-08 11:43:55.889848: I tensorflow/stream_executor/stream.cc:4787] [stream=0x26908340,impl=0x269083e0] did not memcpy device-to-host; source: 0x7f6f384e8800
/usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer22GetElapsedMillisecondsEv+0x97)[0x7f494aa78507]
[f1f0059c329e:60773] [ 5] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer12MicrosecondsEv+0x9)[0x7f494a9e1959]
[f1f0059c329e:60773] [ 6] 2019-03-08 11:43:55.890299: I tensorflow/stream_executor/stream.cc:4787] [stream=0x26908340,impl=0x269083e0] did not memcpy device-to-host; source: 0x7f6f38226400
2019-03-08 11:43:55.890981: I tensorflow/stream_executor/stream.cc:4825] [stream=0x26908340,impl=0x269083e0] did not memzero GPU location; source: 0x7f706cfb4c00
2019-03-08 11:43:55.891007: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7f706cfb4bf0
2019-03-08 11:43:55.891024: I tensorflow/stream_executor/stream.cc:1826] [stream=0x26908340,impl=0x269083e0] did not enqueue 'start timer': 0x7f706cfb4bf0
2019-03-08 11:43:55.891044: I tensorflow/stream_executor/stream.cc:1838] [stream=0x26908340,impl=0x269083e0] did not enqueue 'stop timer': 0x7f706cfb4bf0
2019-03-08 11:43:55.891052: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
[f1f0059c329e:60777] *** Process received signal ***
[f1f0059c329e:60777] Signal: Aborted (6)
[f1f0059c329e:60777] Signal code:  (-6)
/usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(_ZN10tensorflow10BiasGradOpIN5Eigen9GpuDeviceENS1_4halfEE7ComputeEPNS_15OpKernelContextE+0x33a)[0x7f494eb74d5a]
[f1f0059c329e:60773] [ 7] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice13ComputeHelperEPNS_8OpKernelEPNS_15OpKernelContextE+0x48a)[0x7f494a553a6a]
[f1f0059c329e:60773] [ 8] [f1f0059c329e:60777] [ 0] /lib/x86_64-linux-gnu/libpthread.so.0(+0x11390)[0x7f7195040390]
[f1f0059c329e:60777] [ 1] /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x38)[0x7f719458a428]
[f1f0059c329e:60777] [ 2] /lib/x86_64-linux-gnu/libc.so.6(abort+0x16a)[0x7f719458c02a]
[f1f0059c329e:60777] [ 3] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice7ComputeEPNS_8OpKernelEPNS_15OpKernelContextE+0x2a)[0x7f494a55478a]
[f1f0059c329e:60773] [ 9] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dbe0)[0x7f494a5aabe0]
[f1f0059c329e:60773] [10] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dc6f)[0x7f494a5aac6f]
[f1f0059c329e:60773] [11] 2019-03-08 11:43:55.892251: I tensorflow/stream_executor/stream.cc:4787] [stream=0x26908340,impl=0x269083e0] did not memcpy device-to-host; source: 0x7f6f38c20f00
2019-03-08 11:43:55.892299: I tensorflow/stream_executor/stream.cc:4787] [stream=0x26908340,impl=0x269083e0] did not memcpy device-to-host; source: 0x7f6f38c20f00
2019-03-08 11:43:55.892364: I tensorflow/stream_executor/stream.cc:4787] [stream=0x26908340,impl=0x269083e0] did not memcpy device-to-host; source: 0x7f6f38c20f00
2019-03-08 11:43:55.892414: I tensorflow/stream_executor/stream.cc:4787] [stream=0x26908340,impl=0x269083e0] did not memcpy device-to-host; source: 0x7f6f32420300
/usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN5Eigen15ThreadPoolTemplIN10tensorflow6thread16EigenEnvironmentEE10WorkerLoopEi+0x2e2)[0x7f494a639c72]
[f1f0059c329e:60773] [12] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(+0x6c5ce04)[0x7f70dfa8ee04]
[f1f0059c329e:60777] [ 4] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNSt17_Function_handlerIFvvEZN10tensorflow6thread16EigenEnvironment12CreateThreadESt8functionIS0_EEUlvE_E9_M_invokeERKSt9_Any_data+0x48)[0x7f494a636e68]
[f1f0059c329e:60773] [13] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xb8c80)[0x7f495dc4dc80]
[f1f0059c329e:60773] [14] /lib/x86_64-linux-gnu/libpthread.so.0(+0x76ba)[0x7f4a0720c6ba]
[f1f0059c329e:60773] [15] 2019-03-08 11:43:55.892749: I tensorflow/stream_executor/stream.cc:4825] [stream=0x26908340,impl=0x269083e0] did not memzero GPU location; source: 0x7f7059ffdc00
2019-03-08 11:43:55.892761: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7f7059ffdbf0
2019-03-08 11:43:55.892767: I tensorflow/stream_executor/stream.cc:1826] [stream=0x26908340,impl=0x269083e0] did not enqueue 'start timer': 0x7f7059ffdbf0
/lib/x86_64-linux-gnu/libc.so.6(clone+0x6d)[0x7f4a0683241d]
[f1f0059c329e:60773] *** End of error message ***
2019-03-08 11:43:55.892781: I tensorflow/stream_executor/stream.cc:1838] [stream=0x26908340,impl=0x269083e0] did not enqueue 'stop timer': 0x7f7059ffdbf0
2019-03-08 11:43:55.892787: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
2019-03-08 11:43:55.962449: E tensorflow/stream_executor/cuda/cuda_blas.cc:694] failed to run cuBLAS routine cublasSgemmEx: CUBLAS_STATUS_EXECUTION_FAILED
2019-03-08 11:43:55.965064: I tensorflow/stream_executor/stream.cc:4825] [stream=0x27230bb0,impl=0x27230c50] did not memzero GPU location; source: 0x7f500cffbc00
2019-03-08 11:43:55.965116: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7f500cffbbf0
2019-03-08 11:43:55.965125: I tensorflow/stream_executor/stream.cc:1826] [stream=0x27230bb0,impl=0x27230c50] did not enqueue 'start timer': 0x7f500cffbbf0
2019-03-08 11:43:55.965148: I tensorflow/stream_executor/stream.cc:1838] [stream=0x27230bb0,impl=0x27230c50] did not enqueue 'stop timer': 0x7f500cffbbf0
2019-03-08 11:43:55.965159: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
[f1f0059c329e:60776] *** Process received signal ***
[f1f0059c329e:60776] Signal: Aborted (6)
[f1f0059c329e:60776] Signal code:  (-6)
[f1f0059c329e:60776] [ 0] /lib/x86_64-linux-gnu/libpthread.so.0(+0x11390)[0x7f514767a390]
[f1f0059c329e:60776] [ 1] /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x38)[0x7f5146bc4428]
[f1f0059c329e:60776] [ 2] /lib/x86_64-linux-gnu/libc.so.6(abort+0x16a)[0x7f5146bc602a]
[f1f0059c329e:60776] [ 3] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(+0x6c5ce04)[0x7f50920dee04]
[f1f0059c329e:60776] [ 4] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer22GetElapsedMillisecondsEv+0x97)[0x7f508aef2507]
[f1f0059c329e:60776] [ 5] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer12MicrosecondsEv+0x9)[0x7f508ae5b959]
[f1f0059c329e:60776] [ 6] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(_ZN10tensorflow10BiasGradOpIN5Eigen9GpuDeviceENS1_4halfEE7ComputeEPNS_15OpKernelContextE+0x33a)[0x7f508efeed5a]
[f1f0059c329e:60776] [ 7] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice13ComputeHelperEPNS_8OpKernelEPNS_15OpKernelContextE+0x48a)[0x7f508a9cda6a]
[f1f0059c329e:60776] [ 8] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice7ComputeEPNS_8OpKernelEPNS_15OpKernelContextE+0x2a)[0x7f508a9ce78a]
[f1f0059c329e:60776] [ 9] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dbe0)[0x7f508aa24be0]
[f1f0059c329e:60776] [10] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dc6f)[0x7f508aa24c6f]
[f1f0059c329e:60776] [11] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN5Eigen15ThreadPoolTemplIN10tensorflow6thread16EigenEnvironmentEE10WorkerLoopEi+0x2e2)[0x7f508aab3c72]
[f1f0059c329e:60776] [12] 2019-03-08 11:43:55.970074: I tensorflow/stream_executor/stream.cc:4825] [stream=0x27230bb0,impl=0x27230c50] did not memzero GPU location; source: 0x7f500c7fac00
2019-03-08 11:43:55.970106: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7f500c7fabf0
2019-03-08 11:43:55.970113: I tensorflow/stream_executor/stream.cc:1826] [stream=0x27230bb0,impl=0x27230c50] did not enqueue 'start timer': 0x7f500c7fabf0
2019-03-08 11:43:55.970128: I tensorflow/stream_executor/stream.cc:1838] [stream=0x27230bb0,impl=0x27230c50] did not enqueue 'stop timer': 0x7f500c7fabf0
2019-03-08 11:43:55.970135: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
-------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
-------------------------------------------------------
2019-03-08 11:43:56.136354: E tensorflow/stream_executor/cuda/cuda_blas.cc:694] failed to run cuBLAS routine cublasSgemmEx: CUBLAS_STATUS_EXECUTION_FAILED
2019-03-08 11:43:56.136907: I tensorflow/stream_executor/stream.cc:4787] [stream=0x27108880,impl=0x27108920] did not memcpy device-to-host; source: 0x7fbb200e3700
2019-03-08 11:43:56.136966: I tensorflow/stream_executor/stream.cc:4787] [stream=0x27108880,impl=0x27108920] did not memcpy device-to-host; source: 0x7fbb200e3700
2019-03-08 11:43:56.138165: I tensorflow/stream_executor/stream.cc:4825] [stream=0x27108880,impl=0x27108920] did not memzero GPU location; source: 0x7fbc58fb4c00
2019-03-08 11:43:56.138190: I tensorflow/stream_executor/stream.cc:315] did not allocate timer: 0x7fbc58fb4bf0
2019-03-08 11:43:56.138201: I tensorflow/stream_executor/stream.cc:1826] [stream=0x27108880,impl=0x27108920] did not enqueue 'start timer': 0x7fbc58fb4bf0
2019-03-08 11:43:56.138291: I tensorflow/stream_executor/stream.cc:1838] [stream=0x27108880,impl=0x27108920] did not enqueue 'stop timer': 0x7fbc58fb4bf0
2019-03-08 11:43:56.138300: F tensorflow/stream_executor/gpu/gpu_timer.cc:65] Check failed: start_event_ != nullptr && stop_event_ != nullptr
[f1f0059c329e:60774] *** Process received signal ***
[f1f0059c329e:60774] Signal: Aborted (6)
[f1f0059c329e:60774] Signal code:  (-6)
[f1f0059c329e:60774] [ 0] /lib/x86_64-linux-gnu/libpthread.so.0(+0x11390)[0x7fbd80f35390]
[f1f0059c329e:60774] [ 1] /lib/x86_64-linux-gnu/libc.so.6(gsignal+0x38)[0x7fbd8047f428]
[f1f0059c329e:60774] [ 2] /lib/x86_64-linux-gnu/libc.so.6(abort+0x16a)[0x7fbd8048102a]
[f1f0059c329e:60774] [ 3] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(+0x6c5ce04)[0x7fbccb983e04]
[f1f0059c329e:60774] [ 4] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer22GetElapsedMillisecondsEv+0x97)[0x7fbcc4797507]
[f1f0059c329e:60774] [ 5] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNK15stream_executor3gpu8GpuTimer12MicrosecondsEv+0x9)[0x7fbcc4700959]
[f1f0059c329e:60774] [ 6] /usr/local/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so(_ZN10tensorflow10BiasGradOpIN5Eigen9GpuDeviceENS1_4halfEE7ComputeEPNS_15OpKernelContextE+0x33a)[0x7fbcc8893d5a]
[f1f0059c329e:60774] [ 7] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice13ComputeHelperEPNS_8OpKernelEPNS_15OpKernelContextE+0x48a)[0x7fbcc4272a6a]
[f1f0059c329e:60774] [ 8] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN10tensorflow13BaseGPUDevice7ComputeEPNS_8OpKernelEPNS_15OpKernelContextE+0x2a)[0x7fbcc427378a]
[f1f0059c329e:60774] [ 9] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dbe0)[0x7fbcc42c9be0]
[f1f0059c329e:60774] [10] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(+0x77dc6f)[0x7fbcc42c9c6f]
[f1f0059c329e:60774] [11] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZN5Eigen15ThreadPoolTemplIN10tensorflow6thread16EigenEnvironmentEE10WorkerLoopEi+0x2e2)[0x7fbcc4358c72]
[f1f0059c329e:60774] [12] /usr/local/lib/python3.6/site-packages/tensorflow/python/../libtensorflow_framework.so(_ZNSt17_Function_handlerIFvvEZN10tensorflow6thread16EigenEnvironment12CreateThreadESt8functionIS0_EEUlvE_E9_M_invokeERKSt9_Any_data+0x48)[0x7fbcc4355e68]
[f1f0059c329e:60774] [13] /usr/lib/x86_64-linux-gnu/libstdc++.so.6(+0xb8c80)[0x7fbcd796cc80]
[f1f0059c329e:60774] [14] /lib/x86_64-linux-gnu/libpthread.so.0(+0x76ba)[0x7fbd80f2b6ba]
[f1f0059c329e:60774] [15] /lib/x86_64-linux-gnu/libc.so.6(clone+0x6d)[0x7fbd8055141d]
[f1f0059c329e:60774] *** End of error message ***
--------------------------------------------------------------------------
mpirun.real noticed that process rank 1 with PID 0 on node f1f0059c329e exited on signal 6 (Aborted).
--------------------------------------------------------------------------
```


### 8 GPU, bs=4 fp16
```
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/tensorpack-mask-rcnn/MaskRCNN/train.py", line 657, in <module>
    launch_train_with_config(traincfg, trainer)
  File "/tensorpack-mask-rcnn/tensorpack/train/interface.py", line 94, in launch_train_with_config
    extra_callbacks=config.extra_callbacks)
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 343, in train_with_defaults
    steps_per_epoch, starting_epoch, max_epoch)
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 315, in train
    self.main_loop(steps_per_epoch, starting_epoch, max_epoch)
  File "/tensorpack-mask-rcnn/tensorpack/utils/argtools.py", line 176, in wrapper
    return func(*args, **kwargs)
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 280, in main_loop
    self.run_step()  # implemented by subclass
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 180, in run_step
    self.hooked_sess.run(self.train_op)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 694, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1189, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1288, in run
    raise six.reraise(*original_exc_info)
  File "/usr/local/lib/python3.6/site-packages/six.py", line 693, in reraise
    raise value
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1273, in run
    return self._sess.run(*args, **kwargs)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1345, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1109, in run
    return self._sess.run(*args, **kwargs)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 930, in run
    run_metadata_ptr)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1153, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1329, in _do_run
    run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1349, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed : a.shape=(9633792, 1), b.shape=(1, 4), m=9633792, n=4, k=1
	 [[node fpn/fpn/upsample_lat3/Tensordot/MatMul (defined at tensorpack-mask-rcnn/tensorpack/models/pool.py:130) ]]
	 [[gradients/GatherV2_23_grad/Shape/_6257]]

Original stack trace for 'fpn/fpn/upsample_lat3/Tensordot/MatMul':
  File "tensorpack-mask-rcnn/MaskRCNN/train.py", line 657, in <module>
    launch_train_with_config(traincfg, trainer)
  File "tensorpack-mask-rcnn/tensorpack/train/interface.py", line 84, in launch_train_with_config
    model._build_graph_get_cost, model.get_optimizer)
  File "tensorpack-mask-rcnn/tensorpack/utils/argtools.py", line 176, in wrapper
    return func(*args, **kwargs)
  File "tensorpack-mask-rcnn/tensorpack/train/tower.py", line 215, in setup_graph
    train_callbacks = self._setup_graph(input, get_cost_fn, get_opt_fn)
  File "tensorpack-mask-rcnn/tensorpack/train/trainers.py", line 410, in _setup_graph
    grads = self._make_get_grad_fn(input, get_cost_fn, get_opt_fn)()
  File "tensorpack-mask-rcnn/tensorpack/train/tower.py", line 284, in get_grad_fn
    return compute_grad_from_inputs(*inputs)
  File "tensorpack-mask-rcnn/tensorpack/train/tower.py", line 246, in compute_grad_from_inputs
    cost = get_cost_fn(*inputs)
  File "tensorpack-mask-rcnn/tensorpack/tfutils/tower.py", line 286, in __call__
    output = self._tower_fn(*args)
  File "tensorpack-mask-rcnn/tensorpack/graph_builder/model_desc.py", line 262, in _build_graph_get_cost
    ret = self.build_graph(*inputs)
  File "tensorpack-mask-rcnn/MaskRCNN/train.py", line 124, in build_graph
    features = self.backbone(images)
  File "tensorpack-mask-rcnn/MaskRCNN/train.py", line 193, in backbone
    p23456 = fpn_model('fpn', c2345, fp16=self.fp16)
  File "tensorpack-mask-rcnn/tensorpack/models/registry.py", line 128, in wrapped_func
    outputs = func(*args, **actual_args)
  File "tensorpack-mask-rcnn/MaskRCNN/model_fpn.py", line 80, in fpn_model
    lat = lat + upsample2x('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1])
  File "tensorpack-mask-rcnn/MaskRCNN/model_fpn.py", line 57, in upsample2x
    data_format='channels_first')
  File "tensorpack-mask-rcnn/tensorpack/models/registry.py", line 128, in wrapped_func
    outputs = func(*args, **actual_args)
  File "tensorpack-mask-rcnn/tensorpack/models/pool.py", line 130, in FixedUnPooling
    ret = tf.tensordot(x, mat, axes=1)  # bxcxhxwxshxsw
  File "usr/local/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py", line 3641, in tensordot
    ab_matmul = matmul(a_reshape, b_reshape)
  File "usr/local/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py", line 2513, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "usr/local/lib/python3.6/site-packages/tensorflow/python/ops/gen_math_ops.py", line 5675, in mat_mul
    name=name)
  File "usr/local/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 800, in _apply_op_helper
    op_def=op_def)
  File "usr/local/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "usr/local/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 3473, in create_op
    op_def=op_def)
  File "usr/local/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1961, in __init__
    self._traceback = tf_stack.extract_stack()

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1335, in _do_call
    return fn(*args)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1320, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1408, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed : a.shape=(9633792, 1), b.shape=(1, 4), m=9633792, n=4, k=1
	 [[{{node fpn/fpn/upsample_lat3/Tensordot/MatMul}}]]
	 [[gradients/GatherV2_23_grad/Shape/_6257]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/tensorpack-mask-rcnn/MaskRCNN/train.py", line 657, in <module>
    launch_train_with_config(traincfg, trainer)
  File "/tensorpack-mask-rcnn/tensorpack/train/interface.py", line 94, in launch_train_with_config
    extra_callbacks=config.extra_callbacks)
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 343, in train_with_defaults
    steps_per_epoch, starting_epoch, max_epoch)
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 315, in train
    self.main_loop(steps_per_epoch, starting_epoch, max_epoch)
  File "/tensorpack-mask-rcnn/tensorpack/utils/argtools.py", line 176, in wrapper
    return func(*args, **kwargs)
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 280, in main_loop
    self.run_step()  # implemented by subclass
  File "/tensorpack-mask-rcnn/tensorpack/train/base.py", line 180, in run_step
    self.hooked_sess.run(self.train_op)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 694, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1189, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1288, in run
    raise six.reraise(*original_exc_info)
  File "/usr/local/lib/python3.6/site-packages/six.py", line 693, in reraise
    raise value
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1273, in run
    return self._sess.run(*args, **kwargs)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1345, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1109, in run
    return self._sess.run(*args, **kwargs)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 930, in run
    run_metadata_ptr)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1153, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1329, in _do_run
    run_metadata)
  File "/usr/local/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1349, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed : a.shape=(9633792, 1), b.shape=(1, 4), m=9633792, n=4, k=1
	 [[node fpn/fpn/upsample_lat3/Tensordot/MatMul (defined at tensorpack-mask-rcnn/tensorpack/models/pool.py:130) ]]
	 [[gradients/GatherV2_23_grad/Shape/_6257]]
```