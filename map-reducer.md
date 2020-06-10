## Using Map Reducer with Python: Choosing What Type of Wine to Produce

### Business Problem
A new Californian vineyard wants to explore the business opportunity for creating a new exciting wine flavor. Before the company decides on what wine flavor to create, they want to understand who their competition is and how many different wine flavors are currently being produced. The company does not want to make a wine flavor that is currently over saturated in their market, so they are conducting research on what wine flavors are not being created as much. The company also wants to ensure that there will be enough people that is interested in the wine flavor that they choose before they begin a costly batch process of creating the new flavor. 

### Data Used
The data that was used from an excel spreadsheet from Kaggle that included different information from a WineEnthusiast website. Originally the data set had included 14 columns consisting of row number, country of origin, review description, vineyard where the wine was made, total quality points reviewed from a scale of 1-100, price, vineyard state it was made, state region, state region 2, taster name, taster twitter name, title of review, type of wine and the name of winery that made it. The data was cleaned to remove rows that had empty values and was eventually reduced to the most important columns of country, vineyard, score, price, state, wine type and company name. 

### Solving with Map Reducer
This business problem was solved by using MapReduce and setting the wine type as the key and country, region, brand and review score as the values. Once the key was mapped with all the different values, it was important to reduce the values that only included the relevant competition within their area. In the reducer stage, country was filtered by only companies in the U.S., region was filtered to only competitors in California, competitors were only reduced if they had exceeded at least 10 reviews making them a considerable threat, and the average taste score among those competitors were only included if they had at least an average score over 80. After the filtering process, the companies were sorted alphabetically, and the wine type was yielded with the relevant values as well as the count of how many companies produced that wine type. It was important to keep a running count of competitors that produced the different wine flavors even if they didn’t meet the filtered criteria so the company could still understand how large that individual wine flavor’s market was.

### Code
```python

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRTotalWine(MRJob): # good
    def steps(self):
        return [
            MRStep(mapper=self.mapper,

                   reducer=self.reducer)]
    
    def mapper(self, key, line): # good
        (country, vineyard, score, price, region, winetype, brand ) = line.split(',') # split each column
        yield winetype, (country, region, brand, float(score)) # set winetype to key and gather the country, region, brand and taste score for the values

    def reducer(self, key, value):
        count = 0 # set count and vsum set them to zero to calculate the taste score average later
        vsum = 0
        newlist = [] # create an empty list to store our filtered and values
        for country, region, brand, score in sorted(value):
            count += 1
            vsum += score
            scoreaverage = vsum/count
            for brd in sorted(brand): # sort the list alphabetically by the brand when filtering is done
                if country == "US": # only filter the countries that are from the US
                    if count >= 10: # only include the values in the winetype if they have been reviewed at least 10 times
                        if region == "California": # we only want to include the regions that are in California, US
                            if scoreaverage > 80: # if the average score is greater than 80 then include it
                                newlist.append((round(scoreaverage), brand, region, country)) # append all of the filtered and score averages to the empty list
                            
        yield key, (count, newlist) # inlcude the winetype as the key and include the count of how many different vineyards produce that wine, include new list

            
     

if __name__ == '__main__':
    MRTotalWine.run()
    
    
#!python AssignmentPart3.py winedata.csv > assignmentpart3.txt

```

### Results
After running the MapReduce algorithm there was informative findings that would help the new wine company limit its risk while yielding a profitable return. There are wine flavors like Garganega and Pinot Bianco that had over 70 reviews but, neither of which had a company that produced more than 10 reviews meaning that there was demand for the flavor but, no one company had a high market share in it. There was also an abundant amount of companies that produced red wine flavors that tells the company that this wine flavor is overly saturated and should stay away from creating that type of wine. After looking at the data, the company should have multiple opportunities to penetrate the market by making underdeveloped wine flavors.


