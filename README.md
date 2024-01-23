
### References
1. https://www.statsmodels.org/devel/_modules/statsmodels/tsa/stattools.html
2. https://github.com/QuantConnect/Tutorials
3. https://gist.github.com/ThGreatExplorer/ce3f69f810d86a8cf9dc12e87d79a967
4. https://link.springer.com/book/10.1007/978-1-4419-6876-0 (Specifically Chapters 2,4,5, and 7. Read Chapter 3 to review Probability and Statistics)

# Data Collection
[Bloomberg Methodology](Bonds.docx)

## Data
- [BBG Excels](Data/BBG_Excels/)
- [CSV](Data/csv/)

# Statistical Arbitrage in Federal Bond Markets
By Daniel Yu and Nezam Jazayeri 

## Abstract
By nature, arbitrage opportunities are transient and concentrated in underlooked sectors. This research was motivated by the prevalence and history of trading models built on statistical arbitrage techniques in more traditional, liquid financial markets namely public equities, commodities, foreign exchange, and other derivative markets compared to the perceived lack of such exploration in the bond market. Our paper examines Federal Bond markets and indices for statistical arbitrage opportunities across different timescales. We outline the theoretical framework of Bond markets and their differentiation from equity and other financial markets given their unique characteristics such as correlation to federal interest rate, maturity date, yield, etc. Finally, we apply pairs trading techniques and construct basic trading strategies to try and generate alpha from statistical arbitrage methods. We find that ultimately, while the federal bond market is generally efficient, opportunities for statistical arbitrage exist.


## Hypothesis 
Our hypothesis is that Central Banks across the developed world tend to move in tandem, following the US Federal Reserve’s lead, especially demonstrated during the COVID-19 pandemic. The interest rate environment and treasury yields of different countries should move in parallel with macroeconomic trends. Thus, there should exist pair trading opportunities between different government treasuries where we can expect an optimal combination of treasury bond 1 and treasury bond 2 such that if the prices diverge, we can short one and long the other under the expectation that the pair should mean-revert.

See US interest rates on global interest rates:
https://www.imf.org/en/Publications/WP/Issues/2016/12/31/U-S-44315


