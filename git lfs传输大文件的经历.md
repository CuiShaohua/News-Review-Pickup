总结一下最近遇到的git lfs的坑  
==========================
## 1 gitlfs安装  
___________
centos的机子安装git lfs非常容易，直接运行  
```Shell  
yum install gitlfs -y  
```  
## 2 git lfs怎么配合git上传大型文件  
> * 使用track命令： 
```Shell  
git lfs track "relative path"  
```
> * for example, 当前你的仓库绝对路径是/root/project1/，大文件绝对路径是/root/project1/flask/cws.model，然后git lfs就需要写成:
```Shell  
 git lfs track "./flask/cws.model"  
```  
> * 在git add之前先用git lfs track命令，会在仓库文件夹下生成一个.gitattrbute文件，可以查看加入的大文件属性
> * track "relative path"支持正则匹配，例如.model类型的文件可以写成*.model

## 3 git add .   
> * 添加本地仓库文件夹下的所有文件包含文件夹
```Shell  
git add .  
```  
## 5 git commit -m "description of this commit"  
> * 加入本地缓存
```Shell  
git commit -m "description of this commit"  
```  
## 6 git push origin master
> * push之后的origin 和 master分别是本地branch的名称和远程仓库的名称。
```Shell  
git push origin master  
```  

* have fun~
