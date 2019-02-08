using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace BeerML.Regression
{
    public class PriceData
    {
        //[Column(ordinal: "0")]
        //public string FullName;
        //[Column(ordinal: "1")]
        //public float Price;
        //[Column(ordinal: "2")]
        //public float Volume;
        //[Column(ordinal: "3")]
        //public string Type;
        //[Column(ordinal: "4")]
        //public string Country;

        [Column(ordinal: "0")]
        public int Year;
        [Column(ordinal: "1")]
        public int Month;
        [Column(ordinal: "2")]
        public int Day;
        [Column(ordinal: "3")]
        public float Consumption;
        [Column(ordinal: "4")]
        public int Companies;
        [Column(ordinal: "5")]
        public int Installations;
    }

    public class PricePrediction
    {
        //[ColumnName("Score")]
        //public float Price;

        [ColumnName("Score")]
        public float Consumption;
    }

    public class RegressionDemo
    {
        public static void Run()
        {
            // Define context
            var mlContext = new MLContext(seed: 0);

            // Define data file format
            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    //new TextLoader.Column("FullName", DataKind.Text, 0),
                    //new TextLoader.Column("Price", DataKind.R4, 1),
                    //new TextLoader.Column("Volume", DataKind.R4, 2),
                    //new TextLoader.Column("Type", DataKind.Text, 3),
                    //new TextLoader.Column("Country", DataKind.Text, 4)

                    new TextLoader.Column("Year", DataKind.I4, 0),
                    new TextLoader.Column("Month", DataKind.I4, 1),
                    new TextLoader.Column("Day", DataKind.I4, 2),
                    new TextLoader.Column("Consumption", DataKind.R4, 3),
                    new TextLoader.Column("Companies", DataKind.I4, 4),
                    new TextLoader.Column("Installations", DataKind.I4, 5)
                }
            });

            // Load training data
            //var trainingDataView = textLoader.Read("3_Regression/problem3_train.csv");
            var trainingDataView = textLoader.Read("3_Regression/consumption_train.csv");
            // Define features
            //var dataProcessPipeline = mlContext.Transforms.CopyColumns("Price", "Label")
            //                .Append(mlContext.Transforms.Text.FeaturizeText("FullName", "FullNameFeaturized"))
            //                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Type", "TypeEncoded"))
            //                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Country", "CountryEncoded"))
            //                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Volume", "VolumeEncoded"))
            //                .Append(mlContext.Transforms.Concatenate("Features", "FullNameFeaturized", "TypeEncoded", "CountryEncoded", "VolumeEncoded"));

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Consumption", "Label")
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Year", "YearEncoded"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Month", "MonthEncoded"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Day", "DayEncoded"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Companies", "CompaniesEncoded"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Installations", "InstallationsEncoded"))
                                .Append(mlContext.Transforms.Concatenate("Features", "YearEncoded", "MonthEncoded", "DayEncoded", "CompaniesEncoded", "InstallationsEncoded"));


            // Use Poisson Regressionn
            var trainer = mlContext.Regression.Trainers.PoissonRegression(labelColumn: "Label", featureColumn: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);


            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            // Use model for predictions
            IEnumerable<PriceData> drinks = new[]
            {
                new PriceData { Year=2018, Month=01, Day=01, Companies=9, Installations=58 },
                new PriceData { Year=2018, Month=01, Day=02, Companies=9, Installations=58 },
                new PriceData { Year=2018, Month=01, Day=03, Companies=9, Installations=58 },
                new PriceData { Year=2018, Month=01, Day=04, Companies=9, Installations=58 },
                new PriceData { Year=2018, Month=01, Day=05, Companies=9, Installations=58 },
                new PriceData { Year=2018, Month=01, Day=06, Companies=9, Installations=58 },
                new PriceData { Year=2018, Month=01, Day=07, Companies=9, Installations=58 }
                //new PriceData { Year="Hofbräu München Weisse", Type="Öl", Volume=500, Country="Tyskland" },
                //new PriceData { Year="Stefanus Blonde Ale", Type="Öl", Volume=330, Country="Belgien" },
                //new PriceData { Year="Mortgage 10 years", Type="Whisky", Volume=700, Country="Storbritannien" },
                //new PriceData { Year="Mortgage 21 years", Type="Whisky", Volume=700, Country="Storbritannien" },
                //new PriceData { Year="Merlot Classic", Type="Rött vin", Volume=750, Country="Frankrike" },
                //new PriceData { Year="Merlot Grand Cru", Type="Rött vin", Volume=750, Country="Frankrike" },
                //new PriceData { Year="Château de la Berdié Grand Cru", Type="Rött vin", Volume=750, Country="Frankrike" }
            };
//            2018,01,01,690.06503,9,58,
//2018,01,02,1 537.54622,9,58,
//2018,01,03,1 622.22594,9,58,
//2018,01,04,1 582.86276,9,58,
//2018,01,05,1 348.77045,9,58,
//2018,01,06,889.21674,9,58,
//2018,01,07,900.75712,9,58,
//2018,01,08,1 739.73662,9,58,
//2018,01,09,1 764.41514,9,58,
            var predFunction = trainedModel.MakePredictionFunction<PriceData, PricePrediction>(mlContext);

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                Console.WriteLine($"{drink.Day} is {prediction.Consumption}");
            }

            // Evaluate the model
            // var testDataView = textLoader.Read("3_Regression/problem3_validate.csv");
            // var predictions = trainedModel.Transform(testDataView);
            // var metrics = mlContext.Regression.Evaluate(predictions, label: "Label", score: "Score");

        }

    }
}
