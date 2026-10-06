# ROS2-Jazzy及Gazebo-Harmonic安装教程 

因为做别的工程项目需要，所以顺带写了个ROS2-Jazzy及Gazebo-Harmonic安装教程，方便大家快速搭建ROS2开发环境。注意，Jazzy和Gazebo-Harmonic的生态暂时还没有Humble成熟，所以有时候开发没办法像Humble那样有许多开箱即用的包和工具，很多时候需要自己去写一些东西，所以如果你是新手，建议先用Humble开发，等熟悉了ROS2的开发流程后再用Jazzy开发。（例如mid360,D405,D435i的Gazebo仿真插件包官方暂时没有Jazzy版本的）

整个安装过程约一两小时，取决于你的网速
## 安装ROS2-Jazzy 

建议全程打开梯子安装 

1. 更新系统
```bash
sudo apt update
sudo apt upgrade -y
```
2. 设置 UTF-8 环境
```bash
sudo apt install locales -y

sudo locale-gen en_US en_US.UTF-8

sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

export LANG=en_US.UTF-8
```

检查：
```bash
locale
```
3. 添加 ROS 2 软件源

安装工具：

```bash
sudo apt install software-properties-common -y

sudo add-apt-repository universe
```

安装 ROS apt 源：

```bash
sudo apt update

sudo apt install curl -y
```

获取最新 ROS 软件源：

```bash
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
```

下载并安装：
```bash
curl -L -o /tmp/ros2-apt-source.deb \
"https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"

sudo dpkg -i /tmp/ros2-apt-source.deb 
``` 

4. 下载ROS2-Jazzy桌面版（这一步会运行比较久） 
```bash
sudo apt update

sudo apt install ros-jazzy-desktop -y
``` 

5. 安装开发工具(用于创建工作空间、编译节点)： 
```bash
sudo apt install ros-dev-tools -y
``` 
6. 配置ROS2环境
当前终端生效：
```bash
source /opt/ros/jazzy/setup.bash
```
永久生效(写到环境里面)：
```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```
重新加载：
```bash
source ~/.bashrc
``` 
然后关闭终端重新打开 

7. 验证安装 

```bash
echo $ROS_DISTRO
``` 
输出：jazzy  

测试节点：

终端1：
```bash
ros2 run demo_nodes_cpp talker
```
终端2：
```bash
ros2 run demo_nodes_py listener
```
如果看到：
```bash
Publishing: 'Hello World'
```
说明安装成功。

8. 安装点奇奇怪怪的东西 

用于colcon编译
```bash
sudo apt install python3-colcon-common-extensions -y
``` 

rosdep依赖管理 
```bash
sudo apt install python3-rosdep -y

sudo rosdep init

rosdep update
``` 

常用的ROS2包 

```bash
sudo apt install \
ros-jazzy-navigation2 \
ros-jazzy-nav2-bringup \
ros-jazzy-slam-toolbox \
ros-jazzy-xacro \
ros-jazzy-tf2-tools \
ros-jazzy-rqt \
ros-jazzy-ros2-control \
ros-jazzy-ros2-controllers \
-y
```  

验证rviz2 
```bash
rviz2
``` 
会弹出一个白色带有网格平面的窗口，如果打开报错可能是因为显卡驱动版本不一致或者未安装（按正常是装系统时已经安装好了的），用`nvidia-smi`查看，正常出现这样类似 

```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.xx.xx |
| GPU Name        GeForce RTX 4060 Laptop GPU |
+-----------------------------------------------------------------------------+
``` 

正常情况下驱动不一致重启一下电脑就行了，就能自动匹配了。 

没安装驱动用这个：安装当前推荐驱动。

先查看推荐版本：
```bash
ubuntu-drivers devices
```
输出类似：
```bash
driver : nvidia-driver-580 recommended
```
然后：
```bash
sudo apt update

sudo apt install --reinstall nvidia-driver-580
```
完成后重启：
```bash
sudo reboot
```
验证 tf包 
```bash
ros2 run tf2_tools view_frames
```
正常输出 

```bash
[INFO] [1790522030.292311921] [view_frames]: Listening to tf data for 5.0 seconds...

```

9. 安装git 

见我的算法组教程基础篇配置开发环境章节 

ROS2-jazzy适配的语言是python3.12，系统已自带无需安装，可以直接打开终端输入`python3`查看 

10. 安装Gazebo 

Gazebo Harmonic（Gazebo Sim 8），这是 ROS 2 Jazzy 官方推荐的 Gazebo 版本，注意：与Humble的Gazebo Classic版本不兼容。

## 安装 Gazebo Harmonic 

1. 添加 Gazebo 官方软件源

执行：
```
sudo apt update

sudo apt install curl lsb-release gnupg -y
```
添加 GPG 密钥：
```
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
-o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
```
添加仓库：
```
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
| sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
```
2. 更新软件列表
```
sudo apt update
```
此时应该能看到类似：
```
Get: https://packages.osrfoundation.org/gazebo/ubuntu-stable noble InRelease
```
3. 直接执行：
```bash
sudo apt update

sudo apt install gz-harmonic -y
```
安装完成后检查：
```bash
gz sim --version
```
正常输出类似：
```bash
Gazebo Sim 8.x.x
```
4. 安装 ROS2 Jazzy 与 Gazebo 的桥接包

ROS2 和 Gazebo 通信需要 ros_gz：
```bash
sudo apt install ros-jazzy-ros-gz -y
```
包含：
```
ros_gz_bridge
ros_gz_sim
Gazebo 与 ROS2 Topic 通信接口
```

5. 安装glxinfo
```
sudo apt update

sudo apt install mesa-utils -y 
``` 

6. 测试 Gazebo

启动空世界：
```
gz sim
```
应该打开 Gazebo GUI。然后试着运行第一个人型机器人的案例。 


7. 测试 ROS2 + Gazebo 联动

启动 Gazebo：
```
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=empty.sdf
```
查看节点：
```
ros2 node list
```
应该看到：
```
/gazebo
``` 

常用ROS2和Gazebo Harmonic核心包 
```bash
sudo apt update
sudo apt install \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-parallel-gripper-controller \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-twist-mux \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox \
  ros-jazzy-robot-localization \
  ros-jazzy-pointcloud-to-laserscan \
  ros-jazzy-moveit \
  ros-jazzy-xacro \
  ros-jazzy-tf2-tools \
  ros-jazzy-cv-bridge \
  ros-jazzy-depth-image-proc \
  ros-jazzy-image-geometry \
  ros-jazzy-image-transport-plugins \
  ros-jazzy-rqt-image-view
```
