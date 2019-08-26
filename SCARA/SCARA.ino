#define STEPS 200
#include <Stepper.h>

int X_RESET_PIN = 5;;
int X_STEP_PIN = 4;
int X_DIR_PIN = 3;
int X_ENABLE_PIN = 2;


Stepper x_stepper(STEPS, X_STEP_PIN, X_DIR_PIN);

int rpm_speed = 1; // rev per minute
char buf [20];
char ebuf[20]="";
String str="";
int c=0,k=0;

void flushhbuf()
{
  for(int j=0;j<20;j++)
  {
    buf[j] = ebuf[j];
  }
}

void setup() 
{
  Serial.begin(9600);
  pinMode(X_ENABLE_PIN, OUTPUT);
  pinMode(X_RESET_PIN, OUTPUT);
  digitalWrite(X_ENABLE_PIN, LOW);
  digitalWrite(X_RESET_PIN, HIGH);
}

void loop() 
{
  /*if(Serial.available() > 0)
  {
    str = Serial.readStringUntil('\n');
    delay(20);

    if(str[0] == 'M' && str[1] == '1')
    {
      for(int i = 1; i < str.length(); i++)
      {

        if(str[i+1] == 'Y' && c == 0)
        {
          c++;
          i++;

          x_val = atoi(buf);
          flushhbuf();
          k = -1;
        }
        k++;
      }
      buf[k] += str[i];
    }
  }*/ 
  if(Serial.available() > 0)
  {
    str = Serial.readStringUntil('\n');
    delay(10);

    if(str[0] == 'u')
    {
        rpm_speed = 600;
        digitalWrite(X_ENABLE_PIN, LOW);
        Serial.println("clockwise");
        x_stepper.setSpeed(rpm_speed);
        for(int  i= 0; i < 10000; i++)
        {
          x_stepper.step(1);
        }
        delay(10);
        x_stepper.setSpeed(0);
        digitalWrite(X_ENABLE_PIN, LOW);
    }

    if(str[0] == 'd')
    {
        rpm_speed = 200;
        digitalWrite(X_ENABLE_PIN, LOW);
        Serial.println("CounterClockwise");
        x_stepper.setSpeed(rpm_speed);
        //for(int  i= 0; i < 10000; i++)
        {
          x_stepper.step(-10000);
        }
        delay(10);
        x_stepper.setSpeed(0);
        digitalWrite(X_ENABLE_PIN, LOW);
    }

    if(str[0] == 's')
    {
      Serial.println("Stopped");
      digitalWrite(X_ENABLE_PIN, LOW);
    }
  }
  
}
