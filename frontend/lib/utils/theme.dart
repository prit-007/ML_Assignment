import 'package:flutter/material.dart';

ThemeData buildAppTheme() {
  // Primary color palette - Teal/Green
  final primaryColor = Color(0xFF06B6D4); // Vibrant teal
  final primaryVariant = Color(0xFF0891B2); // Darker teal
  final secondaryColor = Color(0xFFEAB308); // Yellow for accents

  // Background and surface colors
  final backgroundColor = Color(0xFFF9FAFB);
  final surfaceColor = Colors.white;
  final errorColor = Color(0xFFEF4444);

  return ThemeData(
    // Base colors
    primaryColor: primaryColor,
    colorScheme: ColorScheme(
      primary: primaryColor,
      primaryContainer: primaryVariant,
      secondary: secondaryColor,
      secondaryContainer: secondaryColor.withOpacity(0.8),
      surface: surfaceColor,
      background: backgroundColor,
      error: errorColor,
      onPrimary: Colors.white,
      onSecondary: Colors.black,
      onSurface: Color(0xFF111827),
      onBackground: Color(0xFF111827),
      onError: Colors.white,
      brightness: Brightness.light,
    ),

    // Scaffold background color
    scaffoldBackgroundColor: backgroundColor,
    cardColor: surfaceColor,

    // Text theme
    textTheme: TextTheme(
      headlineLarge: TextStyle(
        color: Color(0xFF111827),
        fontWeight: FontWeight.bold,
        fontSize: 28,
      ),
      headlineMedium: TextStyle(
        color: Color(0xFF111827),
        fontWeight: FontWeight.bold,
        fontSize: 24,
      ),
      headlineSmall: TextStyle(
        color: Color(0xFF111827),
        fontWeight: FontWeight.w600,
        fontSize: 20,
      ),
      titleLarge: TextStyle(
        color: Color(0xFF111827),
        fontWeight: FontWeight.w600,
        fontSize: 18,
      ),
      bodyLarge: TextStyle(
        color: Color(0xFF374151),
        fontSize: 16,
      ),
      bodyMedium: TextStyle(
        color: Color(0xFF374151),
        fontSize: 14,
      ),
    ),

    // AppBar theme
    appBarTheme: AppBarTheme(
      backgroundColor: primaryColor,
      elevation: 0,
      centerTitle: true,
      iconTheme: IconThemeData(color: Colors.white),
      titleTextStyle: TextStyle(
        color: Colors.white,
        fontSize: 20,
        fontWeight: FontWeight.w600,
      ),
    ),

    // Card theme
    cardTheme: CardTheme(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      shadowColor: Colors.black.withOpacity(0.08),
    ),

    // Button themes
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: primaryColor,
        foregroundColor: Colors.white,
        elevation: 1,
        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
      ),
    ),

    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: primaryColor,
        side: BorderSide(color: primaryColor),
        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
      ),
    ),

    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: primaryColor,
        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
    ),

    // Input decoration
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: Colors.grey[50],
      contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: Colors.grey[300]!, width: 1),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: Colors.grey[300]!, width: 1),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: primaryColor, width: 2),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: BorderSide(color: errorColor, width: 1),
      ),
      errorStyle: TextStyle(color: errorColor),
    ),

    // Divider theme
    dividerTheme: DividerThemeData(
      color: Colors.grey[200],
      thickness: 1,
      space: 24,
    ),

    // Icon theme
    iconTheme: IconThemeData(
      color: Color(0xFF6B7280),
      size: 24,
    ),

    // Progress indicator theme
    progressIndicatorTheme: ProgressIndicatorThemeData(
      color: primaryColor,
      circularTrackColor: Colors.grey[200],
    ),
  );
}
