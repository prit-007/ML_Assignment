class AnalysisResult {
  final DataInsights insights;
  final ModelRecommendation modelRecommendation;

  AnalysisResult({
    required this.insights,
    required this.modelRecommendation,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      insights: DataInsights.fromJson(json['insights'] as Map<String, dynamic>),
      modelRecommendation: ModelRecommendation.fromJson(
          json['model_recommendation'] as Map<String, dynamic>),
    );
  }
}

class DataInsights {
  final int rows;
  final int columns;
  final int missingValues;
  final double missingPercentage;
  final int categoricalFeatures;
  final int numericalFeatures;
  final String targetColumn;
  final Map<String, dynamic>? classDistribution;
  final List<HighCorrelation> highCorrelations;
  final Map<String, NumericalStat> numericalStats;
  final List<String> preprocessingSuggestions;

  DataInsights({
    required this.rows,
    required this.columns,
    required this.missingValues,
    required this.missingPercentage,
    required this.categoricalFeatures,
    required this.numericalFeatures,
    required this.targetColumn,
    this.classDistribution,
    required this.highCorrelations,
    required this.numericalStats,
    required this.preprocessingSuggestions,
  });

  factory DataInsights.fromJson(Map<String, dynamic> json) {
    // Handle high correlations
    List<HighCorrelation> correlations = [];
    if (json['high_correlations'] != null) {
      for (var item in json['high_correlations'] as List) {
        correlations
            .add(HighCorrelation.fromJson(item as Map<String, dynamic>));
      }
    }

    // Handle numerical stats
    Map<String, NumericalStat> numStats = {};
    if (json['numerical_stats'] != null) {
      (json['numerical_stats'] as Map<String, dynamic>).forEach((key, value) {
        numStats[key] = NumericalStat.fromJson(value as Map<String, dynamic>);
      });
    }

    // Handle preprocessing suggestions
    List<String> suggestions = [];
    if (json['preprocessing_suggestions'] != null) {
      suggestions =
          List<String>.from(json['preprocessing_suggestions'] as List);
    }

    return DataInsights(
      rows: json['rows'] as int,
      columns: json['columns'] as int,
      missingValues: json['missing_values'] as int,
      missingPercentage: (json['missing_percentage'] as num).toDouble(),
      categoricalFeatures: json['categorical_features'] as int,
      numericalFeatures: json['numerical_features'] as int,
      targetColumn: json['target_column'] as String,
      classDistribution: json['class_distribution'] as Map<String, dynamic>?,
      highCorrelations: correlations,
      numericalStats: numStats,
      preprocessingSuggestions: suggestions,
    );
  }
}

class HighCorrelation {
  final String feature1;
  final String feature2;
  final double correlation;

  HighCorrelation({
    required this.feature1,
    required this.feature2,
    required this.correlation,
  });

  factory HighCorrelation.fromJson(Map<String, dynamic> json) {
    return HighCorrelation(
      feature1: json['feature1'] as String,
      feature2: json['feature2'] as String,
      correlation: (json['correlation'] as num).toDouble(),
    );
  }
}

class NumericalStat {
  final double min;
  final double max;
  final double mean;
  final double median;
  final double std;

  NumericalStat({
    required this.min,
    required this.max,
    required this.mean,
    required this.median,
    required this.std,
  });

  factory NumericalStat.fromJson(Map<String, dynamic> json) {
    return NumericalStat(
      min: (json['min'] as num).toDouble(),
      max: (json['max'] as num).toDouble(),
      mean: (json['mean'] as num).toDouble(),
      median: (json['median'] as num).toDouble(),
      std: (json['std'] as num).toDouble(),
    );
  }
}

class ModelRecommendation {
  final String problemType;
  final String bestModel;
  final double score;
  final Map<String, ModelResult> allResults;
  final double trainingTime;
  final Map<String, String>? metricsExplanation;
  final List<String> tips;

  ModelRecommendation({
    required this.problemType,
    required this.bestModel,
    required this.score,
    required this.allResults,
    required this.trainingTime,
    this.metricsExplanation,
    required this.tips,
  });

  factory ModelRecommendation.fromJson(Map<String, dynamic> json) {
    Map<String, ModelResult> results = {};

    if (json['all_results'] != null) {
      (json['all_results'] as Map<String, dynamic>).forEach((key, value) {
        if (value['trained'] == true) {
          results[key] = ModelResult.fromJson(value as Map<String, dynamic>);
        }
      });
    }

    return ModelRecommendation(
      problemType: json['problem_type'] as String,
      bestModel: json['best_model'] as String,
      score: (json['score'] as num).toDouble(),
      allResults: results,
      trainingTime: (json['training_time'] as num).toDouble(),
      metricsExplanation: json['metrics_explanation'] != null
          ? Map<String, String>.from(
              json['metrics_explanation'] as Map<String, dynamic>)
          : null,
      tips: json['tips'] != null ? List<String>.from(json['tips'] as List) : [],
    );
  }
}

class ModelResult {
  final bool trained;
  final Map<String, double> metrics;
  final Map<String, double>? featureImportance;
  final String explanation;

  ModelResult({
    required this.trained,
    required this.metrics,
    this.featureImportance,
    required this.explanation,
  });

  factory ModelResult.fromJson(Map<String, dynamic> json) {
    // Handle metrics
    Map<String, double> metricValues = {};
    if (json['metrics'] != null) {
      (json['metrics'] as Map<String, dynamic>).forEach((key, value) {
        metricValues[key] = (value as num).toDouble();
      });
    }

    // Handle feature importance
    Map<String, double>? importanceValues;
    if (json['feature_importance'] != null) {
      importanceValues = {};
      (json['feature_importance'] as Map<String, dynamic>)
          .forEach((key, value) {
        importanceValues![key] = (value as num).toDouble();
      });
    }

    return ModelResult(
      trained: json['trained'] as bool,
      metrics: metricValues,
      featureImportance: importanceValues,
      explanation: json['explanation'] as String? ?? '',
    );
  }
}
