table <- read.csv('~/Projects/cordonflow/roads_accu.csv',header = FALSE)
table <- table[order(table$V1),]

plot(c(20,100), c(0,1), type="n", xlab = "Time interval (seconds)", ylab="Accuracy")
lines(table$V1,table$V2, type = "b", pch=15)
lines(table$V1,table$V3, type = "b", pch=16)
legend(40,0.95, c("Without network information","With network information"), cex = 0.8, pch=c(15,16), lty =1)

