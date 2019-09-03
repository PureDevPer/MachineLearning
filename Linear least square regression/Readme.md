# Linear least square regression

Fitting polynomials with linear least squares. Given an input variable x and a target variable y, our goal is to fit a polynomial function of degree d, as 𝑓(𝑥)=𝑎<sub>𝑑</sub><sup>𝑥</sup> +𝑎<sub>𝑑−1</sub><sup>𝑥</sup> +⋯+𝑎<sub>1</sub><sup>𝑥</sup>+𝑎<sub>0</sub>,to minimize $\frac{1}{m}$$\Sigma$(𝑦<sub>i</sub> −𝑓(𝑥<sub>i</sub>))<sup>2</sup>.

1. the file LLS.dat contains the input variable x and target y as a text file, each row is one data point. [hint: text file can be read using python command open and then convert to numerical values.]

2. Implement least squares regression to fit polynomials of degree d = 1, 3, 5 and 7 to this data set. Your code should output the parameters of the polynomial and the optimal fitting error.

3. Plot using python package matplotlib the each of the regression results in a separate figure, specifying the order of the model in the title of the figure. In each figure, first plot the raw data, your axes should be x and y, corresponding to the input and target variables. Then in the same figure, plot the output of the regression function obtained for the input data using a different symbol.
