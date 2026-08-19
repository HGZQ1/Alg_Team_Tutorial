# 广美视觉组培训之一：双系统安装及ubuntu深度学习环境配置（windows+ubuntu） 
  __作者：小琴羽喵 18927743799__  
  本人github：https://github.com/HGZQ1  
  __时间：2026.01.29__  
  __广州美术学院 艺创RC2026视觉组__  
  >本项目仅用于学习，有误处请指正，希望能看懂孩子们  

__对于装双系统这个事并不要觉得很难，ubuntu系统是视觉组的同学们必须要学会使用的操作系统，可能大多数同学都是用习惯了win，对于一个新的操作系统可能会有抗拒和害怕，但是ubuntu系统只要深入了解和学习，你将会发现它其实操作起来并不比win难，重要就是跳出win的舒适圈。__  


## 一，双系统安装ubuntu22.04（外接移动硬盘当存储）
[__Ubuntu__](https://baike.baidu.com/item/Ubuntu/155795) 是基于 Debian 的开源 Linux 发行版，提供桌面版和服务器版，适合开发者、运维人员及 Linux 新手。而且ROS什么的都得用ubuntu来搞，且大部分视觉小电脑搭载的都是ubuntu系统所以这个系统还是建议大家都得掌握。 __请务必按照步骤操作，否则可能会出现各种问题__  
> 这里选择外接硬盘来存储Ubuntu数据其实也是为了避免出现与windows共盘而导致的一些bug，而且这个切换系统也很方便，一拔一插的事

## 1. 准备工作(硬件)
  - 1.一个空闲的U盘（建议8G以上）✅
  - 2.一个空闲的硬盘（建议500G以上）✅
  - 3.一台要装置Ubuntu的电脑✅
  - 4.一颗不耐烦的心😄(可选)
  - <img src=.\picture\20260129172743_453_14.png alt="U盘和硬盘" width="100" height="100">


## 2. 准备工作(软件)
  - 1.一个Ubuntu系统镜像（建议18.04以上）✅我这边装的是22.04LTS版本，建议视觉组的同学装统一版本22.04
  - 2.一个制作Ubuntu系统启动盘工具（建议Rufus开源免费）✅我用的是UltraISO来制作启动盘(思考...)
  - 3.一个硬盘刷写工具（我用的DiskGenius）✅
  ### 2.1 如果不知道ubuntu系统镜像在哪，看这里！
  官网：https://releases.ubuntu.com/jammy/ubuntu-22.04.5-desktop-amd64.iso  
  清华源：https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.5-desktop-amd64.iso （__如果官网链接下载慢用国内这个__ ✨）  
  阿里源：https://mirrors.aliyun.com/ubuntu-releases/22.04/ubuntu-22.04.5-desktop-amd64.iso  
  中科大源：https://mirrors.ustc.edu.cn/ubuntu-releases/22.04.5/ubuntu-22.04.5-desktop-amd64.iso
  ### 2.2 如果不知道制作启动盘工具在哪，看这里！
  __rufus__ 官网：https://rufus.ie/zh/  
  下载地址：https://github.com/pbatard/rufus/releases/download/v3.15/rufus-3.15.exe  
  UltimaISO：https://www.ultraiso.com/zh/download.html (__试用版就行，其他要付费__)
  ### 2.3 如果不知道硬盘刷写工具在哪，看这里！
  官网：https://www.diskgenius.cn/  
  下载地址：https://www.diskgenius.cn/download.html
## 3.电脑配置  
在安装前，需准备 64 位 CPU、4GB 内存和 25GB 磁盘空间。  
USB 启动盘（≥4GB）  
## 4.开始安装  
大致会分为这几步：1.制作启动盘 2.制作移动硬盘空盘 3.安装ubuntu系统 4.启动引导修复
### 步骤一  制作启动盘
将 Ubuntu 镜像文件写入 U 盘，制作启动盘。我由于用的UltraISO，所以这里就用UltraISO来制作启动盘演示。（呜呜呜，因为自己装的时候忘记截图了，所以图片大多数是网上找的喵~😸✨，如果有机会我可以试试帮组员装系统然后拍多点照片更新一下）
1. 将**U盘**插入电脑。  
2. 打开**UltraISO**。  
   <img src=".\picture\2026-01-29 195315.png" alt="喵" >
3. 点击“**文件**”>“**打开**”按钮，选择 Ubuntu 镜像文件(文件保存在哪就打开哪个)。  
   <img src=".\picture\2026-01-29 195347.png" alt="喵" >  
   <img src=".\picture\2026-01-29 195408.png" alt="喵" >
### 步骤二  写入镜像进盘
1. 成功打开镜像文件后，点击“**启动**”按钮，选择“写入硬盘映像”。  
  
   <img src=".\picture\2026-01-29 200659.png" alt="喵" >  

   **看到硬盘驱动器选项，选择你的U盘(别选错了)**

   <img src=".\picture\屏幕截图 2026-01-29 201119.png" alt="喵" > 

   
   <img src=".\picture\2026-01-29 201056.png" alt="喵" >   

   **映像文件里的就是你的ubuntu系统镜像文件地址，注意核对！！**  
   **U盘里的文件名一定要和镜像文件名一致，否则会出错！！**  

2. 点击“**写入**”按钮，弹出警告，按下“**是**”开始写入，等待写入完成。  
   <img src=".\picture\2026-01-29 203553.png" alt="喵" >  
   **因为在U盘写入时需要将整个U盘清空数据，所以我才建议用空闲的U盘，里面没有文件的U盘。**  
   ✨如果是出现U盘写入时进度条卡住，终止写入并关闭UltraISO，物理拔插U盘，重新打开UltraISO，按照步骤一重新写入U盘。  
   <img src=".\picture\2026-01-29 205239.png" alt="喵" >

3. 写入完成后，点击**关闭**按钮。
4. 然后U盘启动器制作完成！！🚀🚀🚀  

### 步骤三  制作移动硬盘空盘
1. 拔出U盘，插入**移动硬盘**。
2. 打开**DiskGenius**  
   <img src=".\picture\2026-01-29 205537.png" alt="喵" >  
   **选择你的移动硬盘**
   <img src=".\picture\2026-01-29 210152.png" alt="喵" >  
3. 右键你的那个硬盘，选择**转换分区为GUID格式**，**这一步一定要做！！！** ⚠️⚠️⚠️  
   <img src=".\picture\2026-01-29 210720.png" alt="喵" >   
   ✨可能有的U盘型号不同，转换分区为GUID格式的选项为灰色，这说明你的硬盘已经是GUID格式了，可以跳过此步骤。  
4. 刷新一下后再右键你那个硬盘，选择**删除所有分区**，删就完了，因为需要删除分区，这也是为什么我说要找一个空闲的U盘的原因，因为U盘里面有文件，所以不能删除分区。  
   <img src=".\picture\2026-02-01 195645.png" alt="喵" >   
   删除完后上面的蓝色块就会变成灰色，表示该硬盘已经没有分区了。  
5. 然后创建存储盘这一步就搞定了🚀🚀🚀  

### 步骤四  安装Ubuntu系统  
1. **重启电脑**，插入启动盘，进入**BIOS**，将**启动顺序**改为**U盘启动**，F10保存并退出。  (笔者的电脑是拯救者y7000p的，大概在开机出现拯救者logo字样然后按F2就能进入BIOS，其他电脑的启动顺序设置方式可能不同，自行百度喵🐾)
   <img src=".\picture\20260201212444_455_14.png" alt="喵" >  
   <img src=".\picture\20260201212445_456_14.png" alt="喵" >  
   把USB这个选项提到前面去，然后**F10**保存并退出BIOS。然后重启电脑，U盘启动，进入Ubuntu系统启动界面。  
2. 进入ubuntu系统**GRUB启动页面**，它会让你选择启动选项，一般情况下按第一个**Try or Install Ubuntu**,但是如果你像我一样第一个选项启动失败，那么就用第二个选项
**Ubuntu(safe graphics)** 启动，因为这可能是显卡兼容性问题，进入系统后再安装nvidia官方驱动就行了喵。 
   <img src=".\picture\20260201214529_458_14.png" alt="喵" >  
3. 输入**安装**，然后**开始安装**(一般这里应该就没什么问题了，顺着它的选项选就行了)  
4. 然后顺利进入到ubuntu系统桌面✨  
 
   ![alt text](https://res.cloudinary.com/canonical/image/fetch/f_auto,q_auto,fl_sanitize,w_1920/https%3A%2F%2Fassets.ubuntu.com%2Fv1%2Facdf946a-Screenshot%2Bfrom%2B2022-04-18%2B13-05-17.png)  
   
   嗯，一般来说22.04都是这只大水母壁纸(默认)🪼🪼🪼  
   然后就会有一系列的系统基本配置安装引导，就像window一开始引导一样，选语言，设密码，这里就不多赘述了，按照引导进行安装。（如果没有安装引导，一般桌面左上角会有个叫install Ubuntu的图标，点击这个图标就会进入安装引导）  
   <img src=".\picture\2026-02-01 233944.png" alt="喵" >  
   ⚠️⚠️⚠️**注意!!!** ⚠️⚠️⚠️(敲黑板)  
   <img src=".\picture\2026-02-02 000450.png" alt="喵" >  
   **安装类型选择其他选项** ⚠️⚠️⚠️因为你是需要利用移动硬盘来作为该系统的存储设备
   <img src=".\picture\2026-02-02 000507.png" alt="喵" >  
   **更新和其他软件选择如图**，一般这个选项会安装闭源的显卡驱动，适配你电脑显卡，以及一些无线网卡驱动，通常较新的电脑都建议勾选。  
5. **注意，准备分盘** 📝📝📝注意听课宝宝们！  
   选择所需安装的硬盘，一般刚刚你制作好的空硬盘下面会写**空闲**二字，不过你也可以通过看看每个盘有多少存储来判断，这里是**MB**为单位 ， **/dev/sda** 为硬盘名称，如果电脑存在多个硬盘，那么还会存在其他名称，如 /dev/nvme0、/dev/nvme1 等，名称下方为该硬盘的分区情况，点击 " - " 号会删除分区，删除不需要的分区，保证足够大小的空闲分区。  
   1G==1024MB(小知识：10月24日是程序员节)  
   ![alt text](https://i-blog.csdnimg.cn/direct/fecc0776dbfc486f8d4798233300ff4c.png)  这个硬盘就是500G
   
   ---  
   EFI系统分区:  启动的引导分区 efi  
   **建议**：不低于500M（我是2048M） 
   ![alt text](https://i-blog.csdnimg.cn/direct/1264357b57074a1894450ea364c5dd99.jpeg)  
   
   ---  
   交换空间：  swap  
   **建议**：内存大小的两倍（但一般不要超过32G）  
   ![alt text](https://i-blog.csdnimg.cn/direct/5b42022cfbf749f9a59d19d2e90cb5d4.jpeg)  

   ---  

   主分区：  ext4 /  
   **建议**：不低于128G(该分区用于后续配环境CUDNN,CUDA,ROS,Anaconda等等)  
   该分区用于存放系统文件，以及用户数据,linux系统的/目录  
   挂载点：/  
   ![alt text](https://i-blog.csdnimg.cn/direct/5e4d4a638fb1476f80f91d4fb871d220.jpeg)  

   ---  
   
   逻辑分区(可选)：  ext4 /home  
   **建议**：剩余空间(如果不分这个分区，那么系统文件和用户数据都会存放在主分区中，在做上一步**主分区**时可以把剩余存储全放过去)  
   该分区用于存放用户数据，如下载的文件，安装的软件等  
   挂载点：/home  
   ![alt text](https://i-blog.csdnimg.cn/direct/6d072f7678284b15919761d80ffde468.jpeg)  

   ---

   **注意**：  
   1. **EFI系统分区**和**交换空间**必须创建，否则无法启动系统  
   2. **主分区**可以创建多个，但**最多只能创建4个**  
   3. **挂载点**是访问该分区的路径，例如挂载点为/home，则访问该分区时需要输入/home/文件名  
   4. **分区大小**根据个人需求进行设置，但必须保证有足够的空间  
 ---

6. 看到下面的**安装启动器的设备**，选择你的**EFI**分区，一定要保持一致！！！  
   ![alt text](https://i-blog.csdnimg.cn/direct/eb210242b1904908b5f5de53451bc94a.png)  
确认无误后，点击**现在安装**就行了，安装完后会让你继续设置时间，密码，用户名什么什么的，这里就不多说了。  
全部安装好后会需要你重启一遍，重启就是了。  
 <img src=".\picture\2026-02-02 012825.png" alt="喵" > 
 进入了GRUB启动页面，选择**Ubuntu**，进入系统，然后**开始使用Ubuntu**  🚀🚀🚀  后面还差一步就是引导修复了，因为一般直接装载系统的话，windows的引导会被覆盖，所以需要修复一下。  
 
 ---
 ✨✨✨**感谢坚持到这里的你们**✨✨✨  
<figure style="text-align: center;">
  <img src="https://ts2.tc.mm.bing.net/th/id/OIP-C.4dPZxTADMMw5cqyb8SyM7gHaIj?rs=1&pid=ImgDetMain&o=7&rm=3" alt="描述" style="width: 250px; height: 300px;"  />
</figure>




 ---
 ### 步骤五  引导修复  
 1. 进入终端  
  ![alt text](https://img-blog.csdnimg.cn/ccf3b6f4c01a4536a5a19f8c0dfdcbfa.png)  侧边栏也有一个终端图标。  
  输入:(一次输入一个命令(sudo开头)，然后回车,记得联网)  

  ```bash 
  sudo add-apt-repository ppa:yannubuntu/boot-repair
  sudo apt update
  sudo apt-get install boot-repair
  sudo boot-repair
  ```  
 2. 然后它会在扫描完后有个弹窗让你选择修复方式，选择**Recommended repair**，然后等待修复完成即可(如果没有该选项执行多几次```sudo boot-repair```)。  
  <img src=".\picture\2026-02-02 203932.png" alt="喵" >  
  然后reboot重启基本就没什么问题了🎉🎉🎉。  
  然后双系统就搭建好了，需要使用ubuntu时直接插入硬盘就行了，硬盘未插入时会默认打开windows系统，可以开始安装深度学习环境了喵🐾🐾🐾。


  ---
 ### 步骤六  可能会有的一些问题  
 1. 装完ubuntu后，第一次拔掉硬盘切回windows系统可能会进入**Bitlocker恢复界面**（同学们也可以直接在设置里把它关了），按照它说的查找密钥方法，找到密钥后再输入密钥就可以正常进入windows系统了。如果它说启动时找不到硬盘(因为已经把硬盘拔了)的问题，说明它现在只认准了用硬盘来启动系统，可以直接进入BIOS把ubuntu的优先级调到最高(装完ubuntu时会出现这一个神奇的选项的)，windows第二，USB设备第三。  
 2. 步骤四的第二点"进入ubuntu系统**GRUB启动页面**",有的同学可能会启动不了，出现黑屏如下图，可以看看是否是BIOS开启了**Secure Boot**，如果是的话，就关掉它，然后重启电脑，再次进入GRUB启动页面，选择**Try or Install Ubuntu**,然后就可以正常进入系统了。如果还是不行就按**e**进入**Try or Install Ubuntu** 的编辑模式，在**linux**开头那一行的，到**quiet splash**的末尾，删除后面的内容，添加```nomodeset```,基本就没问题了。如果是启动镜像或者U盘问题自己查资料。  
   <img src=".\picture\20260202210925_459_14.jpg" alt="喵" style="width: 300px; height: 200px;"> 
 3. 如果还有什么奇奇怪怪的问题可以私信给作者喵。

---  
参考资料：  
https://www.bilibili.com/video/BV1LP411h7L5?t=165.3    

https://www.bilibili.com/video/BV13wAreeEAF?t=1.4    

https://www.bilibili.com/video/BV1UY4y1z7rd?t=1.0   

https://blog.csdn.net/weixin_44781249/article/details/138048333  


   
---  
## 二, Ubuntu深度学习环境配置(cuda,cudnn,python,anaconda,ros,opencv)  
根据自己需要安装的环境自己添加。  
### 安装Visual Studio Code(vscode微软大战代码)必下✨  
   访问官网下载安装包  https://code.visualstudio.com/download  
<img src=".\picture\2026-02-02 212855.png" alt="喵" >  
下载**linux**的.deb安装包，然后双击安装  
下载好后找到你的安装包的位置，然后右键安装包，选择在**终端中打开** 输入```sudo dpkg -i c```然后按下**Tab**键自动补全，然后回车。  
安装好后输入```code```就会打开vscode了。  
<img src=".\picture\2026-02-02 213919.png" alt="喵" >  