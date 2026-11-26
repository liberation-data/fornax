@echo off
rem The kiln command's launcher, for Windows: the same inputs as bin/kiln (see there).
rem   KILN_JAVA          the java executable, Java 22 or newer (default: %JAVA_HOME%\bin\java.exe, else java on PATH)
rem   KILN_CLI_LIB       the folder of jars (default: ..\lib next to this file)
rem   KILN_LIBRARY_PATH  a folder with the native engine libraries (default: none; a natives jar among the jars)
rem An installer may replace the three @...@ placeholders below with absolute paths.
setlocal

set "KILN_DEFAULT_JAVA=@KILN_JAVA@"
set "KILN_DEFAULT_CLI_LIB=@KILN_CLI_LIB@"
set "KILN_DEFAULT_LIBRARY_PATH=@KILN_LIBRARY_PATH@"
if "%KILN_DEFAULT_JAVA:~0,1%%KILN_DEFAULT_JAVA:~-1%"=="@@" set "KILN_DEFAULT_JAVA="
if "%KILN_DEFAULT_CLI_LIB:~0,1%%KILN_DEFAULT_CLI_LIB:~-1%"=="@@" set "KILN_DEFAULT_CLI_LIB="
if "%KILN_DEFAULT_LIBRARY_PATH:~0,1%%KILN_DEFAULT_LIBRARY_PATH:~-1%"=="@@" set "KILN_DEFAULT_LIBRARY_PATH="

set "JAVA_EXE=%KILN_JAVA%"
if "%JAVA_EXE%"=="" set "JAVA_EXE=%KILN_DEFAULT_JAVA%"
if "%JAVA_EXE%"=="" if not "%JAVA_HOME%"=="" set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
if "%JAVA_EXE%"=="" set "JAVA_EXE=java"

set "LIB=%KILN_CLI_LIB%"
if "%LIB%"=="" set "LIB=%KILN_DEFAULT_CLI_LIB%"
if "%LIB%"=="" set "LIB=%~dp0..\lib"

if "%KILN_LIBRARY_PATH%"=="" set "KILN_LIBRARY_PATH=%KILN_DEFAULT_LIBRARY_PATH%"

"%JAVA_EXE%" --enable-native-access=ALL-UNNAMED %JAVA_OPTS% %KILN_OPTS% -cp "%LIB%\*" org.liberationdata.kiln.cli.Main %*
exit /b %ERRORLEVEL%
