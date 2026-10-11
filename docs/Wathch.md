`watch` 本身不带参数，完整写法就是把命令和刷新间隔拼上：

```bash
watch -n 1 nvidia-smi
```

- `-n 1`：每 **1 秒**刷新一次（默认 2 秒，可写 `-n 0.5`）
- 不指定间隔直接 `watch nvidia-smi` 也行，默认 2 秒

## 常用变体

```bash
# 高亮显示变化的部分（看利用率/显存跳动最直观）
watch -n 1 -d nvidia-smi

# 去掉顶部标题栏，输出更干净
watch -n 1 -t nvidia-smi

# 自定义关键列（推荐，一眼看清显存和利用率）
watch -n 1 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader'
```

## 其实不用 watch 也行

`nvidia-smi` 自带循环模式，效果一样还省事：

```bash
nvidia-smi -l 1          # 每秒刷新一次（-l 2 = 每2秒）
nvidia-smi -lms 500      # 每 500 毫秒
nvidia-smi dmon          # 纯监控视图，只报利用率/显存/功耗，跑训练时最清爽
nvidia-smi dmon -s pucvmet -d 1   # 每秒输出一次，含功耗/温度/显存
```

## 提醒

- 在你的 **WSL** 里同样可用（GPU 直通生效时）。监控训练建议 `nvidia-smi dmon -d 1`，比 `watch` 更省终端刷新开销。
- `watch` 的引号：命令里带空格/管道时**必须用单引号包住**，否则 shell 会拆词报错。

练手时想盯着显存看训练是否 OOM，最顺手的一条：

```bash
watch -n 1 -d 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
```