CC = g++
CFLAGS = -Wall -O2
MAIN = test

test: test.o economy.o agent.o person.o firm.o vectorInFloatOut.o utilMaximizer.o
	$(CC) $(CFLAGS) -o test test.o economy.o agent.o person.o firm.o \
	vectorInFloatOut.o utilMaximizer.o

test.o: test.cpp economy.h
	$(CC) $(CFLAGS) -c test.cpp

economy.o: economy.cpp economy.h
	$(CC) $(CFLAGS) -c economy.cpp

agent.o: agent.cpp economy.h
	$(CC) $(CFLAGS) -c agent.cpp

person.o: person.cpp economy.h
	$(CC) $(CFLAGS) -c person.cpp

firm.o: firm.cpp economy.h
	$(CC) $(CFLAGS) -c firm.cpp

vectorInFloatOut.o: functions/vectorInFloatOut.cpp functions/vectorInFloatOut.h
	$(CC) $(CFLAGS) -c functions/vectorInFloatOut.cpp

utilMaximizer.o: persons/utilMaximizer.cpp economy.h persons/utilMaximizer.h functions/vectorInFloatOut.h
	$(CC) $(CFLAGS) -c persons/utilMaximizer.cpp
