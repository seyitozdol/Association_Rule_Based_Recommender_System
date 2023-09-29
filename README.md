## Association Rule Based Recommender System 
Using the Armut platform's dataset, which captures users and their availed services, we aim to build a recommendation system based on Association Rule Learning.

![image](https://cdn.armut.com/images/og_image_url.jpg)


 **************************
## Business Problem:
Armut, Turkey's largest online service platform, connects service providers with those who seek services. It offers quick and easy access to services such as cleaning, renovation, and transportation with just a few taps on a computer or smartphone. Using a dataset that contains information about the users and the services and categories they have availed, an Association Rule Learning-based recommendation system is desired to be built.
 **************************

 **************************
## Dataset Story:
 The dataset consists of services availed by customers and the categories of these services. It also includes the date and time of each service. The dataset has 4 variables with 162,523 observations and is 5 MB in size.
 **************************

 **************************
## VARIABLES


| Variable Name                        | Description                                    |
| ------------------------------------ |------------------------------------------------|
| UserId          | Customer number                                                                                                                                                 |
| ServiceId       | Anonymized services specific to each category (Example: Service for couch cleaning under the cleaning category). A ServiceId can be found under different categories and might represent different services in those categories. (Example: A service with CategoryId 7 and ServiceId 4 is for radiator cleaning, while a service with CategoryId 2 and ServiceId 4 is for furniture assembly). |
| CategoryId      | Anonymized categories (Example: Cleaning, transportation, renovation)                                                                                            |
| CreateDate      | Date the service was purchased                                                                                                                                  |

## Project Tasks
**Task 1:** Data Prepration

**Task 2:** Generate Association Rules and Provide Recommendations
