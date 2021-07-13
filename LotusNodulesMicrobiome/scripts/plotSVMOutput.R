#open output file from python script
setwd("/Users/mahmoudi/Documents/Doc/2020/DuncanData/")
svH2H2 = read.csv("datafornetwork/LB_All/V1/ML/Result/H2H2_SVMCV5Featureimportance.csv")
file = subset(svH2H2,CV ==1)
file = file[order(file$H2H2),]

file1 = subset(file,!(H2H2 > -6.509193e-03 & H2H2 < 8.682981e-03 ))
sort(file$H2H2,decreasing = T)
file1$Asv = file1$X
file1$Score <- ifelse(file1$H2H2< 0, "H2","H2")
library(ggplot2)
H2H2 = ggplot(data = file1,
              aes(x = reorder(Asv, H2H2), y = H2H2,
                  fill = Score))+
  geom_bar(stat = "identity")+
  coord_flip() + theme_bw() + theme(text = element_text(size = 20))
ggsave("datafornetwork/LB_All/V1/ML/Result/res/H2H2.png",H2H2,width = 7 , height = 10)



