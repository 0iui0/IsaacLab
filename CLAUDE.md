每一次实验都要精细化结构化记录实验过程和结果，有阶段性成果时提交代码  ，记住我们的目标 
0.最终需要合并给isaaclab官方，Marvin  和 cr5af机特定机器人不需要提交到github, 本地保留，确保远端main分支的manager base forge代码干净容易接入其他和frank类似的机器人
1.确保manager based forge和direct forge两个训练的policy表现一致。训练mange base forge要和direct forge的配置机器人资产物理参数完全一致
2.用docker测试确保用isaaclab原有frank机器人资产训练manager base forge pipline没有问题,结果和direct forge训练结果果一致，3.最后用Marvin M6测试训练（urdf需要参考isaaclab已有机器人最佳实践），
4.必要时可以用agent team加速工作
5.wandb api key :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
   wandb account: safezpa, project: forge
   GitHub: git@github.com:0iui0/IsaacLab.git, branch: marvin-m6
6.我需要在github网站看到修改记录和外wandb网站看到训练记录                                                      
.测试训练manager base forge使用rl_games_ppo_cfg.yaml  ，  frank机器人                                                                        
