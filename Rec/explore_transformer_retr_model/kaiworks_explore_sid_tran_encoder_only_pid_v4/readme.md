
256his+128dim

- based on v3

- 这版改了历史序列 用的是colossus里面的1000 然后取最近的256 没有的进行left padding

- 同时添加对应padding位置的mask

- action_cnt 改为3s以上有互动或者大于7s
