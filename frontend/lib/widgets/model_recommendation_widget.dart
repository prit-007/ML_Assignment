import 'package:flutter/material.dart';

import '../models/analysis_result.dart';

class ModelRecommendationWidget extends StatelessWidget {
  final ModelRecommendation recommendation;

  ModelRecommendationWidget({required this.recommendation});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildRecommendationCard(context),
          SizedBox(height: 24),
          _buildPerformanceCard(context),
          SizedBox(height: 24),
          _buildExplanationCard(context),
          SizedBox(height: 24),
          _buildTipsCard(context),
        ],
      ),
    );
  }

  Widget _buildRecommendationCard(BuildContext context) {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Recommended Model',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            Row(
              children: [
                Container(
                  width: 60,
                  height: 60,
                  decoration: BoxDecoration(
                    color: Theme.of(context).primaryColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    Icons.model_training,
                    size: 32,
                    color: Theme.of(context).primaryColor,
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        recommendation.bestModel,
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'Problem Type: ${recommendation.problemType.capitalizeFirst()}',
                        style: TextStyle(
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            SizedBox(height: 16),
            if (recommendation.allResults.containsKey(recommendation.bestModel))
              Text(
                recommendation
                    .allResults[recommendation.bestModel]!.explanation,
                style: TextStyle(fontSize: 14),
              ),
          ],
        ),
      ),
    );
  }

  // widgets/model_recommendation_widget.dart (continued)
  Widget _buildPerformanceCard(BuildContext context) {
    final primaryMetricName =
        recommendation.problemType == 'classification' ? 'accuracy' : 'rmse';
    final scoreLabel = recommendation.problemType == 'classification'
        ? 'Accuracy'
        : 'Root Mean Squared Error';
    final scoreDescription = recommendation.problemType == 'classification'
        ? 'Higher is better'
        : 'Lower is better';

    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Performance Metrics',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: _buildMetricCard(
                    context,
                    scoreLabel,
                    recommendation.problemType == 'classification'
                        ? '${recommendation.score.toStringAsFixed(1)}%'
                        : recommendation.score.toStringAsFixed(3),
                    scoreDescription,
                    recommendation.problemType == 'classification'
                        ? (recommendation.score > 85
                            ? Colors.green
                            : Colors.orange)
                        : (recommendation.score < 1.0
                            ? Colors.green
                            : Colors.orange),
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: _buildMetricCard(
                    context,
                    'Training Time',
                    '${recommendation.trainingTime.toStringAsFixed(2)}s',
                    '',
                    Colors.blue,
                  ),
                ),
              ],
            ),
            if (recommendation.allResults
                    .containsKey(recommendation.bestModel) &&
                recommendation.metricsExplanation != null)
              _buildDetailedMetrics(context),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricCard(BuildContext context, String title, String value,
      String subtitle, Color color) {
    return Container(
      padding: EdgeInsets.symmetric(vertical: 16, horizontal: 12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: 14,
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(
              fontSize: 24,
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
          if (subtitle.isNotEmpty)
            Text(
              subtitle,
              style: TextStyle(
                fontSize: 12,
                color: color.withOpacity(0.7),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildDetailedMetrics(BuildContext context) {
    final modelResult = recommendation.allResults[recommendation.bestModel]!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(height: 16),
        Text(
          'Detailed Metrics',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        SizedBox(height: 8),
        ...modelResult.metrics.entries.map((entry) {
          final explanation =
              recommendation.metricsExplanation![entry.key] ?? '';
          return Padding(
            padding: EdgeInsets.symmetric(vertical: 4),
            child: Row(
              children: [
                Expanded(
                  flex: 2,
                  child: Text(
                    '${entry.key.split('_').map((word) => word.capitalizeFirst()).join(' ')}:',
                  ),
                ),
                Expanded(
                  flex: 1,
                  child: Text(
                    entry.value.toStringAsFixed(4),
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
        if (recommendation.metricsExplanation != null &&
            recommendation.metricsExplanation!.isNotEmpty)
          Padding(
            padding: EdgeInsets.only(top: 8),
            child: Text(
              'Note: ${recommendation.metricsExplanation![recommendation.problemType == 'classification' ? 'accuracy' : 'rmse']}',
              style: TextStyle(
                fontStyle: FontStyle.italic,
                fontSize: 12,
                color: Colors.grey[600],
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildExplanationCard(BuildContext context) {
    if (!recommendation.allResults.containsKey(recommendation.bestModel) ||
        recommendation
            .allResults[recommendation.bestModel]!.explanation.isEmpty) {
      return SizedBox.shrink();
    }

    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Model Explanation',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            Text(
              recommendation.allResults[recommendation.bestModel]!.explanation,
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTipsCard(BuildContext context) {
    if (recommendation.tips.isEmpty) {
      return SizedBox.shrink();
    }

    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Tips for Improvement',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            ...recommendation.tips.map((tip) => Padding(
                  padding: EdgeInsets.only(bottom: 8),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.lightbulb_outline,
                          color: Colors.amber, size: 20),
                      SizedBox(width: 8),
                      Expanded(child: Text(tip)),
                    ],
                  ),
                )),
          ],
        ),
      ),
    );
  }
}

extension StringExtension on String {
  String capitalizeFirst() {
    if (this.isEmpty) return this;
    return this[0].toUpperCase() + this.substring(1);
  }
}
