table <- read.csv('~/Projects/cordonflow/test40/accuracy2.csv',header = FALSE)
table <- table[order(table$V1),]
xrange <- c(0,240)
yrange <- c(0,1)

par(mfrow=c(1,1))
plot(xrange, yrange, type="n", xlab="Time interval (seconds)",ylab = "Accuracy")
lines(table$V1, table$V2,type = "b")

title(main="Flow Estimation Accuracy")

plot(xrange, yrange, type="n", xlab="Time interval (seconds)",ylab = "Heidke skill score")
lines(table$V1, table$V5,type = "b")

title(main="Flow Estimation Skill Score")


par(mfrow= c(1,1))
plot(xrange, yrange, type="n", xlab="Time interval (seconds)", ylab = "Flow Accuracy (fraction correct)")
lines(table$V1, table$V2,type = "b", pch =15)
lines(table$V1, table$V3,type = "b", pch =16)
legend(60,yrange[2], c("LP","Baseline"), cex = 0.8, pch=c(15,16), lty =1)

title(main="Flow Estimation Accuracy")


plot(xrange, c(0,4), type="n", xlab = "Time interval (seconds)", ylab="Avg distance to true destination")
lines(table$V1,table$V8, type = "b", pch=15)
lines(table$V1,table$V9, type = "b", pch=16)
legend(10,4, c("LP","Baseline"), cex = 0.8, pch=c(15,16), lty =1)

title(main="Target Error (distance)")


par(mfrow= c(1,2))

plot(xrange, yrange, type="n", xlab="Time interval (seconds)",ylab = "Accuracy")
lines(table$V1, table$V13,type = "b")

title(main="Flow Distance Accuracy")

plot(xrange, yrange, type="n", xlab="Time interval (seconds)",ylab = "Accuracy")
lines(table$V1, table$V11,type = "b")
title(main="Cell Direction Accuracy (30 Degree)")
