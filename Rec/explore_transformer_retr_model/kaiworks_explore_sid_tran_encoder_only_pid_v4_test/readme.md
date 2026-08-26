
200his+128dim

- based on v4

- 这版改了历史序列 用的是colossus里面的1000

- 筛选中除掉channel为77的，同时channel为37的需要播放大于等于7s，其余的channel都选

- 然后取最近的200 没有的进行left padding

- 同时添加对应padding位置的mask

- action_cnt 改为3s以上有互动或者大于7s
