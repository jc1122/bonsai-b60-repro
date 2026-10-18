#include <level_zero/ze_api.h>
#include <stdlib.h>
ze_result_t zeMemAllocDevice(ze_context_handle_t c,const ze_device_mem_alloc_desc_t*d,size_t n,size_t a,ze_device_handle_t v,void **p) {
    (void)c;(void)d;(void)a;(void)v;*p=calloc(1,n?n:1);return *p?ZE_RESULT_SUCCESS:ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY;
}
ze_result_t zeMemFree(ze_context_handle_t c,void*p){(void)c;free(p);return ZE_RESULT_SUCCESS;}
ze_result_t zeMemFreeExt(ze_context_handle_t c,const ze_memory_free_ext_desc_t*d,void*p){(void)c;(void)d;free(p);return ZE_RESULT_SUCCESS;}
ze_result_t zePhysicalMemCreate(ze_context_handle_t c,ze_device_handle_t v,ze_physical_mem_desc_t*d,ze_physical_mem_handle_t*p){(void)c;(void)v;*p=(ze_physical_mem_handle_t)calloc(1,d->size);return *p?ZE_RESULT_SUCCESS:ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY;}
ze_result_t zePhysicalMemDestroy(ze_context_handle_t c,ze_physical_mem_handle_t p){(void)c;free(p);return ZE_RESULT_SUCCESS;}
