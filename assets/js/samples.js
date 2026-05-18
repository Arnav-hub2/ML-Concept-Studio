(function () {
  window.SampleDatasets = {
    classification: {
      name: "student_success_classification.csv",
      csv: `study_hours,attendance,previous_score,practice_tests,gender,course_level,result
2.5,71,52,1,Female,Foundation,Needs Support
4.0,82,61,2,Male,Foundation,Pass
1.0,58,39,0,Female,Foundation,Needs Support
6.5,91,78,4,Male,Advanced,Pass
5.0,87,70,3,Female,Intermediate,Pass
3.0,75,55,1,Male,Foundation,Needs Support
7.5,95,84,5,Female,Advanced,Pass
2.0,63,45,1,Male,Foundation,Needs Support
4.5,80,66,2,Female,Intermediate,Pass
8.0,96,88,5,Male,Advanced,Pass
3.5,77,59,2,Female,Intermediate,Pass
1.5,61,42,0,Male,Foundation,Needs Support
6.0,89,74,4,Female,Advanced,Pass
2.8,69,50,1,Male,Foundation,Needs Support
5.5,85,72,3,Female,Intermediate,Pass
9.0,98,92,6,Male,Advanced,Pass
4.2,78,64,2,Female,Intermediate,Pass
0.8,55,36,0,Male,Foundation,Needs Support
6.8,92,80,4,Female,Advanced,Pass
3.2,73,57,1,Male,Foundation,Needs Support
7.0,93,83,5,Female,Advanced,Pass
4.8,84,68,3,Male,Intermediate,Pass
2.2,66,47,1,Female,Foundation,Needs Support
5.8,88,75,4,Male,Intermediate,Pass
1.2,59,40,0,Female,Foundation,Needs Support
8.5,97,90,6,Male,Advanced,Pass
3.8,79,62,2,Female,Intermediate,Pass
2.6,68,49,1,Male,Foundation,Needs Support
6.2,90,76,4,Female,Advanced,Pass
4.4,81,65,2,Male,Intermediate,Pass`,
    },
    regression: {
      name: "house_price_regression.csv",
      csv: `area_sqft,bedrooms,age_years,distance_to_city_km,renovated,neighborhood,price_k
850,2,18,12.5,No,East,178
1120,3,12,8.2,Yes,North,268
1450,3,8,6.1,Yes,North,336
980,2,25,15.0,No,South,185
1680,4,6,5.5,Yes,West,412
1250,3,15,9.8,No,East,254
1900,4,4,4.2,Yes,North,498
760,1,28,17.8,No,South,142
1320,3,10,7.4,Yes,West,318
2100,5,3,3.5,Yes,North,552
1180,3,20,11.1,No,East,236
1550,4,9,6.8,No,West,352
920,2,30,18.2,No,South,158
1760,4,7,4.9,Yes,North,446
1010,2,16,10.4,Yes,East,234
2260,5,2,2.8,Yes,West,589
1380,3,11,8.0,No,North,304
860,2,24,16.1,No,South,164
1620,4,5,5.8,Yes,West,401
1200,3,14,9.1,No,East,251
1980,4,6,4.4,Yes,North,510
1080,2,19,12.0,No,East,216
1490,3,7,6.3,Yes,West,361
780,1,32,19.0,No,South,136
1820,4,5,4.7,Yes,North,470
1275,3,13,9.4,No,East,263
2150,5,4,3.9,Yes,West,555
960,2,22,14.2,No,South,177
1525,3,8,6.5,Yes,North,374
1710,4,10,7.2,No,West,386`,
    },
    unsupervised: {
      name: "customer_segments_clustering.csv",
      csv: `annual_income_k,spending_score,visits_per_month,avg_order_value,member_years,channel
32,72,9,38,1,App
35,75,10,41,2,App
28,68,8,34,1,App
82,21,2,118,6,Store
88,18,1,130,7,Store
76,26,3,105,5,Store
55,48,5,64,3,Web
59,52,6,70,4,Web
62,55,6,76,3,Web
24,81,11,29,1,App
91,16,1,142,8,Store
58,50,5,68,4,Web
30,70,9,36,2,App
84,24,2,124,6,Store
64,57,7,78,4,Web
27,77,10,32,1,App
95,19,1,151,9,Store
60,53,6,72,4,Web
33,74,10,39,2,App
80,25,2,112,6,Store
57,46,5,66,3,Web
29,79,11,35,1,App
87,20,1,132,8,Store
61,54,6,73,4,Web
26,83,12,31,1,App
90,17,1,140,8,Store
56,49,5,65,3,Web
34,73,9,40,2,App
78,28,3,108,5,Store
63,56,6,77,4,Web`,
    },
  };
})();
