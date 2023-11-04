# Stage_2A_ENSAE
Stochastic Zeroth-order Optimization in High Dimensions.  

Ce projet est basé sur l'étude empirique de la procédure proposé dans l'article: https://arxiv.org/abs/1710.10551. 
La procédure consiste en l'estimation du gradient d'une fonction, à des entrées en grande dimension disons dans $R^{d}$,dont on ne possède pas une forme explicite de son gradient ni de son gradient, à partir de l'observation de la réalisation bruitées de cette fonction en T points avec  T << p. Ensuite on utilise cet estimation du gradient afin de simuler une descente de gradient stochastique pour estimer l'argmin de cette fonction. 
