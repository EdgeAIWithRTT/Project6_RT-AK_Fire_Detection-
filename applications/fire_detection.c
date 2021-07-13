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

/* fire detection */
#include <rt_ai_fire_model.h>
#include <rt_ai.h>
#include <rt_ai_log.h>
#include <logo.h>
#include <test.h>

struct rt_event ov2640_event;

void ai_run_complete(void *arg){
    *(int*)arg = 1;
}

void bilinera_interpolation(rt_uint8_t *in_array, short height, short width,
                            rt_uint8_t *out_array, short out_height, short out_width);

int fire_app(void){
    lcd_show_image(0, 0, 320, 240, LOGO);
    lcd_show_string(90, 140, 16, "Hello RT-Thread!");
    lcd_show_string(90, 156, 16, "Demo: Fire Detection");
    lcd_show_image(0, 0, 320, 240, TEST);

    rt_err_t result = RT_EOK;
    int ai_run_complete_flag = 0;

    /* find a registered model handle */
    static rt_ai_t model = NULL;
    model = rt_ai_find(RT_AI_FIRE_MODEL_NAME);
    if(!model) {rt_kprintf("ai model find err\r\n"); return -1;}


#if 1
    /* init the model and allocate memory */
    rt_ai_buffer_t *work_buffer = rt_malloc(RT_AI_FIRE_WORK_BUFFER_BYTES+RT_AI_FIRE_IN_TOTAL_SIZE_BYTES+RT_AI_FIRE_OUT_TOTAL_SIZE_BYTES);
    result = rt_ai_init(model, work_buffer);
    if (result != 0) {rt_kprintf("ai model init err\r\n"); return -1;}

    /* prepare input data */
    rt_ai_buffer_t *input_image = rt_malloc(RT_AI_FIRE_IN_1_SIZE_BYTES);
    if (!input_image) {rt_kprintf("malloc input memory err\n"); return -1;}

    // resize
    bilinera_interpolation((rt_uint8_t *)TEST, 240, 320, input_image, 64, 64);

    rt_memcpy(model->input[0], input_image, RT_AI_FIRE_IN_1_SIZE_BYTES);

    /* run ai model */
    result = rt_ai_run(model, ai_run_complete, &ai_run_complete_flag);
    if (result != 0) {rt_kprintf("ai model run err\r\n"); return -1;}

    /* get output and post-process the output */
    uint8_t *pred;
    if(ai_run_complete_flag){
        pred = (uint8_t *)rt_ai_output(model, 0);
        rt_kprintf("prediction: %d %d\n", pred[0], pred[1]);
        // AI_LOG("Prediction: %d\n", prediction);
    }
    rt_free(work_buffer);



    if (pred[0] > 200)
        lcd_show_string(20, 20, 16, "Fire");
    else
        lcd_show_string(20, 20, 16, "No Fire");
#endif
    return 0;
}
MSH_CMD_EXPORT(fire_app,fire demo);
//INIT_COMPONENT_EXPORT(fire_app);


int is_in_array(short x, short y, short height, short width)
{
    if (x >= 0 && x < width && y >= 0 && y < height)
        return 1;
    else
        return 0;
}


void bilinera_interpolation(rt_uint8_t* in_array, short height, short width,
                            rt_uint8_t* out_array, short out_height, short out_width)
{
    double h_times = (double)out_height / (double)height,
           w_times = (double)out_width / (double)width;
    short  x1, y1, x2, y2, f11, f12, f21, f22;
    double x, y;

    for (int i = 0; i < out_height; i++){
        for (int j = 0; j < out_width*3; j=j+3){
            for (int k =0; k <3; k++){
                x = j / w_times + k;
                y = i / h_times;

                x1 = (short)(x - 3);
                x2 = (short)(x + 3);
                y1 = (short)(y + 1);
                y2 = (short)(y - 1);
                f11 = is_in_array(x1, y1, height, width*3) ? in_array[y1*width*3+x1] : 0;
                f12 = is_in_array(x1, y2, height, width*3) ? in_array[y2*width*3+x1] : 0;
                f21 = is_in_array(x2, y1, height, width*3) ? in_array[y1*width*3+x2] : 0;
                f22 = is_in_array(x2, y2, height, width*3) ? in_array[y2*width*3+x2] : 0;
                out_array[i*out_width*3+j+k] = (rt_uint8_t)(((f11 * (x2 - x) * (y2 - y)) +
                                           (f21 * (x - x1) * (y2 - y)) +
                                           (f12 * (x2 - x) * (y - y1)) +
                                           (f22 * (x - x1) * (y - y1))) / ((x2 - x1) * (y2 - y1)));
            }
        }
    }
}
