CC = g++
CFLAGS = -Wall -O2
MAIN = test

test: test.o economy.o agent.o person.o firm.o vecToScalar.o utilMaxer.o
	$(CC) $(CFLAGS) -o test test.o economy.o agent.o person.o firm.o \
	vecToScalar.o utilMaxer.o

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

vecToScalar.o: functions/vecToScalar.cpp functions/vecToScalar.h
	$(CC) $(CFLAGS) -c functions/vecToScalar.cpp

utilMaxer.o: persons/utilMaxer.cpp economy.h persons/utilMaxer.h functions/vecToScalar.h
	$(CC) $(CFLAGS) -c persons/utilMaxer.cpp
