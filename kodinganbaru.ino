#define pwm1 5
#define pwm2 6
#define pwm3 9
#define pwm4 10

// String stringOne; // Initial type of class
int i;

void setup() {
  pinMode(pwm1, OUTPUT);
  pinMode(pwm2, OUTPUT);
  pinMode(pwm3, OUTPUT);
  pinMode(pwm4, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(1);
}

int kiri = 5;
int kanan = 5;

void loop() {
  if (Serial.available() > 0)
  {
    int data;
    i = Serial.readString().toInt();
    if (i == 0){
        if (kiri < 150)
          {
              if (kiri >= 150)
              {
                  kiri = 150;
                  kanan = 150;
              }
              else
              {
                  kiri+=5;
                  kanan+=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
         if (kiri > 150)
          {
              if (kiri <= 155)
              {
                  kiri = 150;
                  kanan = 150;
              }
              else
              {
                  kiri-=5;
                  kanan-=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
    } 
    // else {
      // if (i == 0){
      //   if (kiri < 200)
      //     {
      //         if (kiri >= 195)
      //         {
      //             kiri = 200;
      //             kanan = 200;
      //         }
      //         else
      //         {
      //             kiri+=5;
      //             kanan+=5;
      //         }
      //         analogWrite(pwm1, 0);
      //         analogWrite(pwm2, kiri);
      //         analogWrite(pwm3, 0);
      //         analogWrite(pwm4, kanan);
      //         delay(100); 
      //     }
      //    if (kiri > 200)
      //     {
      //         if (kiri <= 215)
      //         {
      //             kiri = 200;
      //             kanan = 200;
      //         }
      //         else
      //         {
      //             kiri-=5;
      //             kanan-=5;
      //         }
      //         analogWrite(pwm1, 0);
      //         analogWrite(pwm2, kiri);
      //         analogWrite(pwm3, 0);
      //         analogWrite(pwm4, kanan);
      //         delay(100); 
      //     }
//        stringOne = String("Aspal");
      // } 
      else if (i == 1){
        if (kiri < 0)
          {
              if (kiri >= 0)
              {
                  kiri = 0;
                  kanan = 0;
              }
              else
              {
                  kiri+=5;
                  kanan+=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
         if (kiri > 0)
          {
              if (kiri <= 5)
              {
                  kiri = 0;
                  kanan = 0;
              }
              else
              {
                  kiri-=5;
                  kanan-=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
//        stringOne = String("Kerikil");
      } else if (i == 2){
        if (kiri < 155)
          {
              if (kiri >= 145 )
              {
                  kiri = 155;
              }
              else
              {
                  kiri+=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
        if (kanan < 85)
          {
              if (kanan >= 80)
              {
                  kanan = 85;
              }
              else
              {
                  kanan+=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
          if (kiri > 155 )
          {
              if (kiri <= 165 )
              {
                  kiri = 155;
              }
              else
              {
                  kiri-=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
          if (kanan > 85)
          {
              if (kanan <= 90)
              {
                  kanan = 85;
              }
              else
              {
                  kanan-=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
          
        }
     
//        stringOne = String("Lantai");
       else if (i == 3){
        if (kiri < 85)
          {
              if (kiri >= 75 )
              {
                  kiri = 85;
              }
              else
              {
                  kiri+=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
        if (kanan < 155)
          {
              if (kanan >= 145)
              {
                  kanan = 155;
              }
              else
              {
                  kanan+=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
          if (kiri > 85 )
          {
              if (kiri <= 95 )
              {
                  kiri = 85;
              }
              else
              {
                  kiri-=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
          if (kanan > 155)
          {
              if (kanan <= 165)
              {
                  kanan = 155;
              }
              else
              {
                  kanan-=5;
              }
              analogWrite(pwm1, 0);
              analogWrite(pwm2, kiri);
              analogWrite(pwm3, 0);
              analogWrite(pwm4, kanan);
              delay(100); 
          }
//        stringOne = String("Paving");
      }       
    }
    
    delay(100);

    Serial.println(String(kiri) + "+" + String(kanan));
    // Serial.println(String(kanan));
  }
