# Out
OUT = -o ./out/x_ai

# Flags
# -lsqlite3 for lsqlite3
# -lssl -lcrypto for openssl/sha256
# -Wall = Warn ALl
FLAGS = -Wall

# C++ Standard Version
CPPSTDV = -std=c++2a

# File(s)
FILES = ./source/main.cpp

# Default / Main
buildAndRunMain: clear buildMain runMain

# Builds executable
buildMain:
	g++ $(FILES) $(OUT) $(CPPSTDV) $(FLAGS)

# Runs the outputed file
runMain: clear
	./out/x_ai

# Clears The Terminal
clear:
	clear

# Cleans Outs
clean:
	rm ./out/*