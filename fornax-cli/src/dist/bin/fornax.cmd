@echo off
rem The fornax command's launcher, for Windows: the same inputs as bin/fornax (see there).
rem   FORNAX_JAVA          the java executable, Java 22 or newer (default: %JAVA_HOME%\bin\java.exe, else java on PATH)
rem   FORNAX_CLI_LIB       the folder of jars (default: ..\lib next to this file)
rem   FORNAX_LIBRARY_PATH  a folder with the native engine libraries (default: none; a natives jar among the jars)
rem An installer may replace the three @...@ placeholders below with absolute paths.
setlocal

set "FORNAX_DEFAULT_JAVA=@FORNAX_JAVA@"
set "FORNAX_DEFAULT_CLI_LIB=@FORNAX_CLI_LIB@"
set "FORNAX_DEFAULT_LIBRARY_PATH=@FORNAX_LIBRARY_PATH@"
if "%FORNAX_DEFAULT_JAVA:~0,1%%FORNAX_DEFAULT_JAVA:~-1%"=="@@" set "FORNAX_DEFAULT_JAVA="
if "%FORNAX_DEFAULT_CLI_LIB:~0,1%%FORNAX_DEFAULT_CLI_LIB:~-1%"=="@@" set "FORNAX_DEFAULT_CLI_LIB="
if "%FORNAX_DEFAULT_LIBRARY_PATH:~0,1%%FORNAX_DEFAULT_LIBRARY_PATH:~-1%"=="@@" set "FORNAX_DEFAULT_LIBRARY_PATH="

set "JAVA_EXE=%FORNAX_JAVA%"
if "%JAVA_EXE%"=="" set "JAVA_EXE=%FORNAX_DEFAULT_JAVA%"
if "%JAVA_EXE%"=="" if not "%JAVA_HOME%"=="" set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
if "%JAVA_EXE%"=="" set "JAVA_EXE=java"

set "LIB=%FORNAX_CLI_LIB%"
if "%LIB%"=="" set "LIB=%FORNAX_DEFAULT_CLI_LIB%"
if "%LIB%"=="" set "LIB=%~dp0..\lib"

if "%FORNAX_LIBRARY_PATH%"=="" set "FORNAX_LIBRARY_PATH=%FORNAX_DEFAULT_LIBRARY_PATH%"

"%JAVA_EXE%" --enable-native-access=ALL-UNNAMED %JAVA_OPTS% %FORNAX_OPTS% -cp "%LIB%\*" org.liberationdata.fornax.cli.Main %*
exit /b %ERRORLEVEL%
