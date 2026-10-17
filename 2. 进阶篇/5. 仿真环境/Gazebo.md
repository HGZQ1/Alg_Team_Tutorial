# Gazebo 

## 仿真 
**仿真简单来说就是用计算机里的模型和程序去模仿真实系统或过程的行为。** 对于机器人来说就是在虚拟环境里面“造”一个虚拟的机器人和虚拟的场景，让算法像在真机上一样运行、测试。 

那么仿真有什么用呢：
  - 没有真机也能开发和调试算法（例如机械一直不出车，算法该如何调试和测试代码呢）
  - 提供安全的测试环境，通常上真机进行测试算法时你也不确定你的算法是否能运行，是否能正常工作，如果直接上机测试说不好会失控或者到处乱创，到时候修复成本就有点高了，所以建议更新完一版算法后自己先在仿真里面跑几次再部署到真机（回归测试）。
  - 快速迭代，有时候部署新一版到真机上也是挺麻烦的，所以通常都是更新几版或者两版后再部署到真机上，其中我们就可以用仿真来测试我们的每一版算法
  - 生成训练数据用于强化学习
  

ROS2本身是不具备仿真功能的。所以需要仿真工具需要额外下载。而Gazebo对ROS2的支持会更加友好。 

## 关于仿真软件选择 
MuJoCo 的设计目标是轻量、快速、适合强化学习大规模采样；Isaac Sim 的目标是高保真渲染加 GPU 并行下的具身任务开发；Gazebo 的目标则是搭在 ROS 生态里的系统级机器人仿真。目标不同，底层取舍自然不同。

##  Gazebo Classic  

Gazbeo Classic对于的配置是ubuntu22.04,ROS2 Humble 

Gazebo Harmonic对应的配置是ubuntu24.04，ROS2 Jazzy

<img src="https://ts1.tc.mm.bing.net/th/id/OIP-C.oQyX365oL0umH5xuu18vngHaDn?r=0&rs=1&pid=ImgDetMain&o=7&rm=3" style="zoom:50%;" /> 

安装  

添加源
```bash
sudo sh -c 'echo "deb  http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

```

```bash
sudo apt update
sudo apt install gazebo
``` 

验证安装和启动 
```bash 
gazebo --version
gazebo
```  
安装工具包

```bash
sudo apt install ros-humble-gazebo-ros-pkgs
``` 

Gazebo Classic 是 Gazebo 的传统版本，主要用于机器人仿真。它提供了一个强大的物理引擎和丰富的传感器仿真模型，使得用户可以在虚拟环境中测试和验证机器人算法。你也可以在一些传感器设备官方找到他们的Gazebo仿真插件。 

经典工作流： 

1. 用URDF或SDF描述机器人模型（结构和关节）和环境模型。 
2. 用SDF/world文件描述场景和物理属性。 
3. 启动Gazebo仿真环境，加载机器人和场景模型。
4. ROS2节点与Gazebo进行通信，订阅Gazebo输出的传感器话题，获取传感器数据并控制机器人行为。 
5. 控制指令通过 /cmd_vel 或其他自定义话题发送给Gazebo，驱动机器人运动。 
6. 在Gazebo中观察机器人行为，调试算法，收集数据。  
   
Gazebo通过订阅ROS2话题来获取控制指令，并通过发布话题来输出传感器数据。这样，用户可以在仿真环境中测试和验证机器人算法，而无需实际部署到物理机器人上。
```text 
真机:
   D435i → /camera/depth/image_raw 
                     ↓ 
                   检测包 
                     ↓ 
                   detection_msgs 
                     ↓
                  决策包
                     ↓ 
                   /cmd_vel 
                     ↓ 
                    机器人 

 仿真: 
   
  Gazebo仿真相机 → /camera/depth/image_raw （嗯呐，可以返回仿真环境中相机看到的画面的）
                     ↓ 
                   检测包 
                     ↓ 
                   detection_msgs 
                     ↓
                  决策包
                     ↓ 
                   /cmd_vel 
                     ↓ 
                  仿真机器人
```

## Gazebo Classic GUI 
<img src="https://i-blog.csdnimg.cn/blog_migrate/bf88a4d73eb9195872bfe7359cd8b3a0.png#pic_center" alt="Gazebo GUI" style="zoom:50%;" /> 

使用教程：https://blog.csdn.net/lc1852109/article/details/126082238  

## URDF 
仿真的第一步就是完成对机器人的建模，但是仿真建模跟我们所熟知的blender、maya等等软件不一样，它不是靠网格体去进行变形这样建模，它是需要我们去用特定语言描述机器人的结构和关节的。URDF就是用来描述机器人模型的标准格式。（小知识：其实你们可以鞭挞机械组让他们来完成这一步，因为SW有转URDF插件，但建议转完后把压缩包放到这个网站https://urdf.online/ 来检查他们的URDF包）

**URDF（Unified Robot Description Format）是ROS中用于描述机器人模型的标准格式**。它使用**XML格式**来定义机器人的结构、关节、传感器和其他组件。URDF文件通常包含以下内容：
- 机器人结构：定义机器人的各个部分（如连杆、关节）
- 关节类型：定义关节的类型（如旋转关节、平移关节）和运动范围
- 传感器：定义机器人上安装的传感器（如摄像头、激光雷达）
- 物理属性：定义机器人的质量、惯性和碰撞模型
- 视觉属性：定义机器人的外观和材质 

这里有一个完整的机器人URDF示例，展示了一个简单的移动机器人加六轴机械臂模型，包括底盘、轮子和传感器。https://github.com/HGZQ1/Agricultural-Bot/blob/main/Agricultural_Bot/ros2_ws/src/agri_robot_description/urdf/robot_pick_robot11.urdf （我实习时做的农业采摘仿真机器人） 

<img src="../../picture\屏幕截图 2026-10-17 220252.png" alt="robot" style="width: 260px; height: 300px;">

下面会给出一个简化的URDF示例，展示如何定义一个简单的移动机器人模型：

创建一个robot_description功能包
```bash 
ros2 pkg create robot_description --build-type ament_cmake --license Apache-2.0
```

并在其中创建一个urdf文件夹，然后在urdf文件夹中创建一个名为simple_robot.urdf的文件，输入内容如下：

```xml 
<?xml version="1.0"?> 
<!-- 上面的？xml说的是该文件符合XML1.0规范，是一个XML格式文件-->
<robot name="simple_robot">
  <!-- 开始定义机器人的身体部分 --> 
  <link name="base_link">
  <!-- 部件外观描述 -->
    <visual> 
      <geometry> 
      <!-- 定义一个简单的立方体作为机器人的基础 -->
        <box size="0.5 0.3 0.4"/>
      </geometry> 
      <!-- 定义机器人的颜色和材质 -->
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
  </link> 

    <!-- 定义机器人IMU传感器部分 --> 
    <link name="imu_link">
      <visual>
        <!-- 视觉几何相对于 imu_link 的位置 -->
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <!-- 定义一个简单的圆柱体作为IMU传感器 -->
          <cylinder length="0.2" radius="0.1"/>
        </geometry>
        <material name="orange">
          <color rgba="1 0.4 0 1"/>
        </material>
      </visual>
    </link>

    <!-- 机器人关节定义 -->
    <joint name="base_to_imu" type="fixed">
      <!-- 定义父关节为base_link -->
      <parent link="base_link"/>
      <!-- 定义子关节为imu_link -->
      <child link="imu_link"/>
      <!-- 子部件相对于父部件的平移和旋转，因为不可能所有部件都是恰好安放在base_link这个点上 -->
      <origin xyz="0 0 0.3" rpy="0 0 0"/> 
    </joint>
<!-- 所以上面就定义了一个简单的机器人，它的主要结构是一个立方体，尺寸0.5 0.3 0.1，蓝色；上面搭载了一个圆柱体IMU，高0.02，半径0.05，位于相对于机器人的base_link高0.06位置-->
</robot> 
``` 
然后打开RVIZ,在RVIZ的display模块中找到RobotModel模块，修改Description Source为File，然后在Description File中选择我们的这个文件加载URDF并显示，最上面的改为base_link,这样你就可以看到我们做的机器人模型了。 

<img src="../../picture/Screenshot from 2026-10-18 00-57-47.png"> 


|标签|人话|
|---|---|
|robot |机器人主标签，定义机器人名称
|link	|机器人连杆（刚体）
|joint	|关节（连接两个连杆）
|visual	|视觉模型（定义外观）
|collision	|碰撞形状（用于物理仿真）
|inertial	|惯性信息（质量、惯性矩）
|material	|颜色和材质
|geometry	|形状（球体、立方体、网格等）
|mesh|网格体（可以直接链接到SW模型）


## Xacro 
在原生URDF文件描述复杂机器人时，存在严重的手工维护缺陷，也就是URDF


## SDF 

SDF 由 Gazebo 提出，适用于更复杂的仿真场景。相比 URDF，SDF 能够描述完整的仿真环境，包括多个机器人、传感器、地形等。 

相比于URDF特点：

- 支持完整的物理仿真，包括摩擦、弹性、流体、光照等。
- 适用于 Gazebo、PyBullet、NVIDIA Isaac Sim 等仿真器。
- 支持多个机器人和复杂环境（URDF 只能描述单个机器人）。
- 可以直接在Gazebo中使用，无需转换。
- 采用 XML 结构，类似于 URDF，但功能更丰富。

实际案例：https://github.com/HGZQ1/Agricultural-Bot/blob/main/Agricultural_Bot/sim_ws/src/agri_greenhouse_worlds/models/tomato_0/model.sdf