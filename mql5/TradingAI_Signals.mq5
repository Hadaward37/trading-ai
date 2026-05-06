//+------------------------------------------------------------------+
//|  TradingAI_Signals.mq5                                           |
//|  Reads data/mt5_signals.csv written by core/mt5_signals.py       |
//|  and draws entry arrows + SL/TP lines on the active chart.       |
//|                                                                   |
//|  Install: copy to MT5 MQL5/Experts/ folder, compile and attach   |
//|  to the EURUSD H1 chart. CsvPath default is already set to the   |
//|  correct location: C:\Users\dudut\trading-ai\data\mt5_signals.csv |
//+------------------------------------------------------------------+
#property copyright "trading-ai"
#property version   "1.00"
#property strict

input string CsvPath       = "C:\\Users\\dudut\\trading-ai\\data\\mt5_signals.csv";
input int    RefreshSec    = 15;    // how often to re-read the file (seconds)
input color  BuyColor      = clrLimeGreen;
input color  SellColor     = clrRed;
input color  SlColor       = clrOrangeRed;
input color  TpColor       = clrDodgerBlue;
input int    ArrowSize     = 2;
input string ObjPrefix     = "tai_";

//--- globals
datetime g_last_check = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(RefreshSec);
   LoadAndDraw();
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   DeleteAllObjects();
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   LoadAndDraw();
  }

//+------------------------------------------------------------------+
void DeleteAllObjects()
  {
   int total = ObjectsTotal(0);
   for(int i = total - 1; i >= 0; i--)
     {
      string name = ObjectName(0, i);
      if(StringFind(name, ObjPrefix) == 0)
         ObjectDelete(0, name);
     }
  }

//+------------------------------------------------------------------+
void LoadAndDraw()
  {
   int handle = FileOpen(CsvPath, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
     {
      // try absolute path
      handle = FileOpen(CsvPath, FILE_READ | FILE_CSV | FILE_ANSI, ',');
      if(handle == INVALID_HANDLE)
        {
         Print("TradingAI: cannot open ", CsvPath, " error=", GetLastError());
         return;
        }
     }

   DeleteAllObjects();

   // skip header
   string line[];
   if(!FileIsEnding(handle))
      FileReadArray(handle, line, 0, 9);

   int count = 0;
   while(!FileIsEnding(handle))
     {
      if(!FileReadArray(handle, line, 0, 9)) break;
      if(ArraySize(line) < 9) continue;

      // CSV: datetime, symbol, timeframe, signal, entry, sl, tp, score, atr
      string dt_str   = line[0];
      string symbol   = line[1];
      // line[2] = timeframe
      int    sig      = (int)StringToInteger(line[3]);
      double entry    = StringToDouble(line[4]);
      double sl       = StringToDouble(line[5]);
      double tp       = StringToDouble(line[6]);
      double score    = StringToDouble(line[7]);

      if(sig == 0) continue;

      // Parse datetime string "YYYY-MM-DD HH:MM:SS"
      datetime bar_time = StringToTime(dt_str);

      string prefix   = ObjPrefix + IntegerToString(count) + "_";
      color  clr      = (sig == 1) ? BuyColor : SellColor;
      int    arrow    = (sig == 1) ? 233 : 234;   // ↑ / ↓
      string dir      = (sig == 1) ? "BUY" : "SELL";

      // Entry arrow
      string arr_name = prefix + "arrow";
      ObjectCreate(0, arr_name, OBJ_ARROW, 0, bar_time, entry);
      ObjectSetInteger(0, arr_name, OBJPROP_ARROWCODE, arrow);
      ObjectSetInteger(0, arr_name, OBJPROP_COLOR,     clr);
      ObjectSetInteger(0, arr_name, OBJPROP_WIDTH,     ArrowSize);
      ObjectSetString(0,  arr_name, OBJPROP_TOOLTIP,
                      dir + "  Score:" + DoubleToString(score, 0) + "/100\nEntry:" + DoubleToString(entry, 5));

      // SL line
      string sl_name = prefix + "sl";
      ObjectCreate(0, sl_name, OBJ_TREND, 0, bar_time, sl, bar_time + 86400 * 3, sl);
      ObjectSetInteger(0, sl_name, OBJPROP_COLOR,  SlColor);
      ObjectSetInteger(0, sl_name, OBJPROP_STYLE,  STYLE_DASH);
      ObjectSetInteger(0, sl_name, OBJPROP_WIDTH,  1);
      ObjectSetInteger(0, sl_name, OBJPROP_RAY_RIGHT, false);
      ObjectSetString(0,  sl_name, OBJPROP_TOOLTIP, "SL " + DoubleToString(sl, 5));

      // TP line
      string tp_name = prefix + "tp";
      ObjectCreate(0, tp_name, OBJ_TREND, 0, bar_time, tp, bar_time + 86400 * 3, tp);
      ObjectSetInteger(0, tp_name, OBJPROP_COLOR,  TpColor);
      ObjectSetInteger(0, tp_name, OBJPROP_STYLE,  STYLE_DASH);
      ObjectSetInteger(0, tp_name, OBJPROP_WIDTH,  1);
      ObjectSetInteger(0, tp_name, OBJPROP_RAY_RIGHT, false);
      ObjectSetString(0,  tp_name, OBJPROP_TOOLTIP, "TP " + DoubleToString(tp, 5));

      // Label
      string lbl_name = prefix + "label";
      double lbl_y    = (sig == 1) ? entry - 0.0003 : entry + 0.0003;
      ObjectCreate(0, lbl_name, OBJ_TEXT, 0, bar_time, lbl_y);
      ObjectSetString(0,  lbl_name, OBJPROP_TEXT,
                      dir + " " + DoubleToString(score, 0) + "/100");
      ObjectSetInteger(0, lbl_name, OBJPROP_COLOR,    clr);
      ObjectSetInteger(0, lbl_name, OBJPROP_FONTSIZE, 8);

      count++;
     }

   FileClose(handle);
   ChartRedraw(0);
   Print("TradingAI: drew ", count, " signal(s) from ", CsvPath);
  }
//+------------------------------------------------------------------+
