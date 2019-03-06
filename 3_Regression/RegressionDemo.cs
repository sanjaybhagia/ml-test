using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace BeerML.Regression
{
    public class PriceData
    {
        [Column(ordinal: "3")]
        public int Year;
        [Column(ordinal: "4")]
        public int Month;
        [Column(ordinal: "5")]
        public int Day;
        [Column(ordinal: "7")]
        public float Consumption;
        [Column(ordinal: "6")]
        public int Weekday;
        [Column(ordinal: "8")]
        public float Temperature;
    }

    public class PricePrediction
    {
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
                    new TextLoader.Column("Year", DataKind.I4, 3),
                    new TextLoader.Column("Month", DataKind.I4, 4),
                    new TextLoader.Column("Day", DataKind.I4, 5),
                    new TextLoader.Column("Consumption", DataKind.R4, 7),
                    new TextLoader.Column("Weekday", DataKind.I4, 6),
                    new TextLoader.Column("Temperature", DataKind.R4, 8)
                }
            });

            // Load training data

            var trainingDataView = textLoader.Read("3_Regression/consumption_training.csv");

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Consumption", "Label")
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Year", "YearEncoded"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Month", "MonthEncoded"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Day", "DayEncoded"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Temperature", "TemperatureEncoded"))
                                .Append(mlContext.Transforms.Concatenate("Features", "YearEncoded", "MonthEncoded", "DayEncoded", "TemperatureEncoded")); // "CompaniesEncoded", "InstallationsEncoded"));


            // Use Poisson Regressionn
            var trainer = mlContext.Regression.Trainers.PoissonRegression(labelColumn: "Label", featureColumn: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);


            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            var predFunction = trainedModel.MakePredictionFunction<PriceData, PricePrediction>(mlContext);

            //read evaluation data from csv - consumption_result.csv
            var file = System.IO.File.ReadAllLines("3_Regression/consumption_result.csv");

            List<PriceData> prices = new List<PriceData>();
            var query = from line in file
                        let data = line.Split(',')
                        orderby data[0], data[1]
                        select new
                        {
                            Mother = data[0],
                            Daughter = data[1],
                            PodId = data[2],
                            Year = data[3],
                            Month = data[4],
                            Day = data[5],
                            Weekday = data[6],
                            Consumption = data[7],
                            Temperature = data[8]
                        };
            var dataOnly = query.Skip(1);
            foreach (var s in dataOnly)
            {
                var price = new PriceData()
                {
                    Year = Int32.Parse(s.Year),
                    Month = Int32.Parse(s.Month),
                    Day = Int32.Parse(s.Day),
                    Weekday = Int32.Parse(s.Weekday),
                    Temperature = float.Parse(s.Temperature)
                };
                prices.Add(price);
            }

            using (var w = new StreamWriter("3_Regression/forecast.csv"))
            {
                foreach (var p in prices)
                {
                    var prediction = predFunction.Predict(p);
                    Console.WriteLine($"{p.Year}-{p.Month}-{p.Day} is {prediction.Consumption}");
                    var line = string.Format("{0},{1},{2},{3},{4}", p.Year, p.Month, p.Day, p.Weekday, prediction.Consumption);
                    w.WriteLine(line);
                }
                w.Flush();
            }
        }
    }
}
