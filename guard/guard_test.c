#include <level_zero/ze_api.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
static pthread_barrier_t barrier;
static ze_result_t races[2];
static ze_result_t alloc(size_t n,void **p){return zeMemAllocDevice(NULL,NULL,n,1,(ze_device_handle_t)1,p);}
static void* worker(void *index){size_t i=(size_t)index;void*p=NULL;pthread_barrier_wait(&barrier);races[i]=alloc(80,&p);pthread_barrier_wait(&barrier);if(races[i]==ZE_RESULT_SUCCESS)assert(zeMemFree(NULL,p)==ZE_RESULT_SUCCESS);return NULL;}
int main(void) {
    void*a=NULL,*b=NULL,*c=NULL;
    assert(alloc(64,&a)==ZE_RESULT_SUCCESS);
    assert(alloc(80,&b)==ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY && b==NULL);
    assert(zeMemFree(NULL,a)==ZE_RESULT_SUCCESS);
    assert(alloc(80,&b)==ZE_RESULT_SUCCESS);assert(zeMemFree(NULL,b)==ZE_RESULT_SUCCESS);
    assert(alloc(48,&a)==ZE_RESULT_SUCCESS);assert(alloc(80,&b)==ZE_RESULT_SUCCESS);
    assert(zeMemFreeExt(NULL,NULL,a)==ZE_RESULT_SUCCESS);
    assert(alloc(48,&c)==ZE_RESULT_SUCCESS);assert(zeMemFree(NULL,b)==ZE_RESULT_SUCCESS);assert(zeMemFree(NULL,c)==ZE_RESULT_SUCCESS);
    ze_physical_mem_desc_t d={.size=96};ze_physical_mem_handle_t h=NULL;
    assert(zePhysicalMemCreate(NULL,(ze_device_handle_t)1,&d,&h)==ZE_RESULT_SUCCESS);
    assert(alloc(32,&a)==ZE_RESULT_SUCCESS);assert(alloc(1,&b)==ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(zePhysicalMemDestroy(NULL,h)==ZE_RESULT_SUCCESS);assert(zeMemFree(NULL,a)==ZE_RESULT_SUCCESS);
    pthread_barrier_init(&barrier,NULL,2);pthread_t x,y;pthread_create(&x,NULL,worker,(void*)0);pthread_create(&y,NULL,worker,(void*)1);pthread_join(x,NULL);pthread_join(y,NULL);
    assert((races[0]==ZE_RESULT_SUCCESS)+(races[1]==ZE_RESULT_SUCCESS)==1);
    puts("PASS: allocation limit, free/reuse, extended free, physical allocation, concurrent reservation");return 0;
}
