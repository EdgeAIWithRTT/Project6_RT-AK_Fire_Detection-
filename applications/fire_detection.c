/*
 * @Author: lebhoryi@gmail.com
 * @Date: 2021-06-30 16:52:58
 * @LastEditors: lebhoryi@gmail.com
 * @LastEditTime: 2021-07-01 19:36:20
 * @Version: V0.0.1
 * @FilePath: /art_pi/applications/fire_detection.c
 * @Description: fire detection demo app
 */
#include "drv_spi_ili9488.h"  // spi lcd driver
#include <lcd_spi_port.h>  // lcd ports
#include <rt_ai_fire_model.h>
#include <rt_ai.h>
#include <rt_ai_log.h>

/* fire detection */
#include <logo.h>
#include <test.h>

struct rt_event ov2640_event;

int ai_run_complete_flag = 0;
void ai_run_complete(void *arg){
    *(int*)arg = 1;
}

int fire_app(void){
    lcd_show_image(0, 0, 320, 240, LOGO);
    lcd_show_string(90, 140, 16, "Hello RT-Thread!");
    lcd_show_string(90, 156, 16, "Demo: Fire Detection");
    rt_thread_mdelay(1000);
    lcd_clear(BLACK);

    rt_err_t result = RT_EOK;
    static rt_ai_t model = NULL;
    rt_ai_buffer_t *work_buffer = rt_malloc(RT_AI_FIRE_WORK_BUFFER_BYTES+RT_AI_FIRE_IN_TOTAL_SIZE_BYTES+RT_AI_FIRE_OUT_TOTAL_SIZE_BYTES);

    // find a registered model handle
    model = rt_ai_find(RT_AI_FIRE_MODEL_NAME);
    if(!model) {rt_kprintf("ai model find err\r\n"); return -1;}

    // init the model and allocate memory
    result = rt_ai_init(model, work_buffer);
    if (result != 0) {rt_kprintf("ai init err\r\n"); return -1;}

    // prepare input data
    rt_memcpy(model->input[0], TEST, RT_AI_FIRE_IN_1_SIZE_BYTES);
    result = rt_ai_run(model, ai_run_complete, &ai_run_complete_flag);
    if (result != 0) {rt_kprintf("ai model run err\r\n"); return -1;}

    // get output and post-process the output
    if(ai_run_complete_flag){
        uint8_t *out = (uint8_t *)rt_ai_output(model, 0);
        rt_kprintf("pred: %d %d\n", out[0], out[1]);
        // AI_LOG("Prediction: %d\n", prediction);
    }
    rt_free(work_buffer);
    return 0;
}
MSH_CMD_EXPORT(fire_app,fire demo);
