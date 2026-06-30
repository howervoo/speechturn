# 数据与流式协议

## 清单

| 字段 | 约定 |
| --- | --- |
| id | 非空、唯一字符串 |
| audio | 音频路径，相对于 JSONL 目录 |
| instruction | 非空文本指令 |
| text | 目标文本，静音样本可为空 |
| speaker / language | 非空标识，默认 unknown / und |
| start / duration | 非负起点、可选正时长，单位秒 |

读取保留清单顺序。按 speaker 分割后，每条数据恰好属于一个集合，集合间说话人互斥。样本不足以形成所请求的非空集合时明确报错。未知字段和重复 ID 不会被静默忽略。

## 文本指标

WER 使用空白分词，CER 使用归一化 Unicode 字符并忽略空白。默认保留大小写与标点；NFKC 和空白处理明确执行。报告包括替换、删除、插入和参考长度，语料级结果先累加编辑数再相除。空参考的分母按 1 处理，避免除零。

## 流式事件

```text
idle --start--> open --final--> closed
                  |
                  +--error--> failed
                  |
                  +--audio/text--> open
```

sequence 从 0 连续递增；timestamp 使用单调时钟，允许相等。非法转换不修改会话。AudioChunker 保持样本顺序，flush 输出余数一次并关闭缓冲区。

first_text_seconds 表示 start 到首次 text 事件的时间；final_seconds 表示 start 到 final；real_time_factor 为处理时长除以输入音频时长。音频时长为零时返回 null。

## 完整性检查

API 快照覆盖每个源码模块；输入测试向量覆盖 SpeechConfig、TrainConfig、AudioExample 和 StreamEvent 的每个序列化字段。流式状态图、小规模文本编辑距离和音频切分执行穷举测试。调整契约必须显式更新快照及对应行为测试。
