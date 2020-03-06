# Predict-the--purchase-probability-of-a-room
 
Problem Description： A room sharing company (e.g., Airbnb) wants to help room providers set a reasonable price for their rooms. One of the key steps is to build a model to predict the  purchase probability of a room (described by certain features as well as the date) under certain  prices. We have the following historic data:

1. ID: The data ID
2. Region: The region the room belongs to (an integer, taking value between 1 and 10)
3. Date: The date of stay (an integer between 1‐365, here we consider only one‐day request)
4. Weekday: Day of week (an integer between 1‐7)
5. Apartment/Room: Whether the room is a whole apartment (1) or just a room (0) 
6. Beds: The number of beds in the room (an integer between 1‐4)
7. Review: Average review of the seller (a continuous variable between 1 and 5)
8. Pic Quality: Quality of the picture of the room (a continuous variable between 0 and 1)
9. Price: The historic posted price of the room (a continuous variable)
10. Accept: Whether this post gets accepted (someone took it, 1) or not (0) in the end

(There are 50,000 training and 20,000 testing data.)

Goal： Build a model to predict the purchase probability of each test data. We will evaluate the model by the AUC of your result (thus please give a probability for each test data), but accuracy, recall and precision will also be evaluated, and decision-making on rendering their importance is part of the test.
