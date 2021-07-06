/*
 * @Author: lebhoryi@gmail.com
 * @Date: 2021-06-30 16:52:58
 * @LastEditors: lebhoryi@gmail.com
 * @LastEditTime: 2021-07-01 19:36:20
 * @Version: V0.0.1
 * @FilePath: /art_pi/applications/fire_detection.c
 * @Description: fire detection demo app
 */

#include <rt_ai_fire_model.h>
#include <rt_ai.h>
#include <rt_ai_log.h>

#include <test.h>  // 64x64
/* fire detection */

int ai_run_complete_flag = 0;
void ai_run_complete(void *arg){
    *(int*)arg = 1;
}

int fire_app(void){
    rt_err_t result = RT_EOK;
    static rt_ai_t model = NULL;
    rt_ai_buffer_t *work_buffer = rt_malloc(RT_AI_FIRE_WORK_BUFFER_BYTES+RT_AI_FIRE_IN_TOTAL_SIZE_BYTES+RT_AI_FIRE_OUT_TOTAL_SIZE_BYTES);

    //find a registered model handle
    model = rt_ai_find(RT_AI_FIRE_MODEL_NAME);
    if(!model) {rt_kprintf("ai model find err\r\n"); return -1;}

    //init the model handle
    result = rt_ai_init(model, work_buffer);
    if (result != 0) {rt_kprintf("ai init err\r\n"); return -1;}

    //prepare input data
    rt_memcpy(model->input[0], TEST, RT_AI_FIRE_IN_1_SIZE_BYTES);
    result = rt_ai_run(model, ai_run_complete, &ai_run_complete_flag);
    if (result != 0) {rt_kprintf("ai model run err\r\n"); return -1;}

    //process the inference data
    if(ai_run_complete_flag){
        //get inference data
        uint8_t *out = (uint8_t *)rt_ai_output(model, 0);
        rt_kprintf("pred: %d %d\n", out[0], out[1]);
        // AI_LOG("Prediction: %d\n", prediction);
    }
    rt_free(work_buffer);
    return 0;
}
MSH_CMD_EXPORT(fire_app, fire detection demo);
