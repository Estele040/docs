cuda 安装

卸载旧的nvidia驱动

```shell
sudo apt-get purge nvidia*
sudo apt-get autoremove

# 禁用nouveau开源驱动（避免冲突）
echo "blacklist nouveau" | sudo tee -a /etc/modprobe.d/blacklist.conf
sudo update-initramfs -u
sudo reboot  # 重启生效
```

打开cuda官网按照命令

```shell
wegt cuda*.run
```

给cuda_*.sh加入权限

```shell
sudo chmod +x cuda_*.sh
```

关闭冲突的进程，临时切换到tty模式（纯命令行模式）

```shell
sudo systemctl isolate multi-user.target
```

安装runfuile包

```shell
sudo ./cuda_*.run
```

安装完成后图形界面

```shell
sudo systemctl isolate graphical.target
```

配置环境变量：编辑`~/,bashrc`或`/etc/profile`

```shell
echo 'export PATH=/usr/local/cuda-*/bin:${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-*/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
```

验证安装：

输入`nvcc -version`显示版本，或运行`nvidia-smi`确认GPU识别正确

## 方法2：使用conda安装

查看conda支持的cuda版本

```shell
conda search cudatoolkit --info
```

执行上述命令后，显示出源内所有cuda版本，以及下载地址。

下载cuda：

```shell
wget cuda_download_link
```

安装cuda：

```shell
conda install --use-local 本地cuda包所在路径
```

查看cuda对应的cudnn版本

```shell
conda search cudnn --info
```

下载cudnn

```shell
conda install --use-local 本地cudnn所在的路径
```

测试安装是否成功：

在conda虚拟环境中想要测试安装是否成功，不能使用`nvcc -V`命令测试，需要在虚拟环境中安装pytorch包进行测试。

测试cuda版本：

```shell
print(torch.version.cuda)
print(torch.backends.cudnn.version())
```

