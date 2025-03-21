import 'package:flutter/material.dart';
import 'package:frontend/screens/home_screen.dart';
import 'package:frontend/utils/theme.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ML Model Analyzer',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: HomeScreen(),
    );
  }
}
