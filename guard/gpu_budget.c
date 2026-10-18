#define _GNU_SOURCE
#include <level_zero/ze_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Process-local allocation accounting. No GPU telemetry or driver changes. */
typedef struct allocation {void *ptr; size_t bytes; int physical; struct allocation *next;} allocation;
static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
static allocation *head;
static size_t budget,live,peak;
__attribute__((constructor)) static void setup(void) {
    const char *value=getenv("BATCH_GPU_BUDGET_BYTES");
    budget=value ? strtoull(value,NULL,10) : 0;
    fprintf(stderr,"GPU_BUDGET enabled budget=%zu bytes\n",budget);
}
__attribute__((destructor)) static void finish(void) {
    fprintf(stderr,"GPU_BUDGET final live=%zu peak=%zu budget=%zu\n",live,peak,budget);
}
static allocation *reserve(size_t bytes,int physical) {
    allocation *item=malloc(sizeof(*item));
    if(!item)return NULL;
    pthread_mutex_lock(&mutex);
    if(bytes>budget || live>budget-bytes) {
        fprintf(stderr,"GPU_BUDGET OUT_OF_DEVICE_MEMORY requested=%zu live=%zu budget=%zu\n",bytes,live,budget);
        pthread_mutex_unlock(&mutex);free(item);return NULL;
    }
    live+=bytes;if(live>peak)peak=live;
    pthread_mutex_unlock(&mutex);
    item->bytes=bytes;item->physical=physical;return item;
}
static void commit(allocation *item,void *ptr,ze_result_t result) {
    pthread_mutex_lock(&mutex);
    if(result==ZE_RESULT_SUCCESS && ptr) {
        item->ptr=ptr;item->next=head;head=item;
        if(item->bytes>=64*1024*1024)
            fprintf(stderr,"GPU_BUDGET allocated=%zu live=%zu peak=%zu\n",item->bytes,live,peak);
    } else {live-=item->bytes;free(item);}
    pthread_mutex_unlock(&mutex);
}
static void release_locked(void *ptr,int physical,ze_result_t result) {
    if(result!=ZE_RESULT_SUCCESS)return;
    allocation **link=&head;
    while(*link) {
        allocation *item=*link;
        if(item->ptr==ptr && item->physical==physical) {
            *link=item->next;live-=item->bytes;free(item);break;
        }
        link=&item->next;
    }
}
ze_result_t ZE_APICALL zeMemAllocDevice(ze_context_handle_t context,
    const ze_device_mem_alloc_desc_t *desc,size_t bytes,size_t alignment,
    ze_device_handle_t device,void **ptr) {
    typedef ze_result_t (ZE_APICALL *fn)(ze_context_handle_t,const ze_device_mem_alloc_desc_t*,size_t,size_t,ze_device_handle_t,void**);
    fn real=(fn)dlsym(RTLD_NEXT,"zeMemAllocDevice");
    if(!real)return ZE_RESULT_ERROR_UNINITIALIZED;
    allocation *item=reserve(bytes,0);
    if(!item){if(ptr)*ptr=NULL;return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;}
    ze_result_t result=real(context,desc,bytes,alignment,device,ptr);
    commit(item,ptr?*ptr:NULL,result);return result;
}
ze_result_t ZE_APICALL zeMemFree(ze_context_handle_t context,void *ptr) {
    typedef ze_result_t (ZE_APICALL *fn)(ze_context_handle_t,void*);
    fn real=(fn)dlsym(RTLD_NEXT,"zeMemFree");
    if(!real)return ZE_RESULT_ERROR_UNINITIALIZED;
    pthread_mutex_lock(&mutex);
    ze_result_t result=real(context,ptr);release_locked(ptr,0,result);
    pthread_mutex_unlock(&mutex);return result;
}
ze_result_t ZE_APICALL zeMemFreeExt(ze_context_handle_t context,
    const ze_memory_free_ext_desc_t *desc,void *ptr) {
    typedef ze_result_t (ZE_APICALL *fn)(ze_context_handle_t,const ze_memory_free_ext_desc_t*,void*);
    fn real=(fn)dlsym(RTLD_NEXT,"zeMemFreeExt");
    if(!real)return ZE_RESULT_ERROR_UNINITIALIZED;
    pthread_mutex_lock(&mutex);
    ze_result_t result=real(context,desc,ptr);release_locked(ptr,0,result);
    pthread_mutex_unlock(&mutex);return result;
}
ze_result_t ZE_APICALL zePhysicalMemCreate(ze_context_handle_t context,
    ze_device_handle_t device,ze_physical_mem_desc_t *desc,ze_physical_mem_handle_t *ptr) {
    typedef ze_result_t (ZE_APICALL *fn)(ze_context_handle_t,ze_device_handle_t,ze_physical_mem_desc_t*,ze_physical_mem_handle_t*);
    fn real=(fn)dlsym(RTLD_NEXT,"zePhysicalMemCreate");
    if(!real)return ZE_RESULT_ERROR_UNINITIALIZED;
    if(!device)return real(context,device,desc,ptr);
    allocation *item=reserve(desc->size,1);
    if(!item){if(ptr)*ptr=NULL;return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;}
    ze_result_t result=real(context,device,desc,ptr);
    commit(item,ptr?(void*)*ptr:NULL,result);return result;
}
ze_result_t ZE_APICALL zePhysicalMemDestroy(ze_context_handle_t context,ze_physical_mem_handle_t ptr) {
    typedef ze_result_t (ZE_APICALL *fn)(ze_context_handle_t,ze_physical_mem_handle_t);
    fn real=(fn)dlsym(RTLD_NEXT,"zePhysicalMemDestroy");
    if(!real)return ZE_RESULT_ERROR_UNINITIALIZED;
    pthread_mutex_lock(&mutex);
    ze_result_t result=real(context,ptr);release_locked((void*)ptr,1,result);
    pthread_mutex_unlock(&mutex);return result;
}
