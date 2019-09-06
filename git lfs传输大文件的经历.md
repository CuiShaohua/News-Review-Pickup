总结一下最近遇到的gitlfs的坑  
==========================
1 gitlfs安装  
___________
centos的机子安装git lfs非常容易，直接运行  
```Shell  
yum install gitlfs -y  
```  
2 git lfs怎么配合git上传大型文件  
>>> * 
```Shell  
git lfs track "relative path"  
```
> * for example, 当前你的仓库绝对路径是/root/project1/，大文件绝对路径是/root/project1/flask/cws.model，然后git lfs就需要写成:
> ```Shell  
\> git lfs track "./flask/cws.model"  
\> ```  
