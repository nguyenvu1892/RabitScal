//+------------------------------------------------------------------+
//| RabitScal_EA.mq5  AI Bridge Client v3.1 (SINGLE SOCKET + MTF)  |
//+------------------------------------------------------------------+
//| Single TCP connection  bidirectional on same socket.             |
//| Sends: TICK (real-time), CANDLES (300 bars  4 TF per symbol)    |
//| Receives: HEARTBEAT, ORDER                                       |
//|                                                                  |
//| Protocol: NEWLINE-DELIMITED (Sep Vu 2026-03-14)                 |
//|   JSON + "\n" one message per line                              |
//|                                                                  |
//| CANDLES payload: 300 bars per timeframe (M1,M5,M15,H1)          |
//| Format: [[unixtime,O,H,L,C,vol], ...]  compact array            |
//+------------------------------------------------------------------+
#property copyright   "RabitScal Team - Sep Vu"
#property version     "3.10"
#property description "AI Bridge v3.1 - Single Socket + MTF 300 Candles"
#property strict

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input string   InpServerIP        = "127.0.0.1";     // Python Server IP
input int      InpPort            = 17777;            // Server port
input int      InpConnectTimeout  = 5000;             // Connect timeout (ms)
input int      InpReadTimeout     = 50;               // Read timeout (ms)
input int      InpHeartbeatMax    = 30;               // Max seconds without heartbeat
input int      InpMagicNumber     = 202603;           // Magic number
input double   InpDefaultLot      = 0.01;             // Default lot size
input int      InpMaxSlippage     = 30;               // Max slippage (points)
input int      InpReconnectDelay  = 5;                // Reconnect delay (seconds)
input int      InpCandleCount     = 300;              // Number of candles to send

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
int            g_socket           = INVALID_HANDLE;
ulong          g_lastHeartbeatMs  = 0;
bool           g_connected        = false;
bool           g_tradingEnabled   = true;
int            g_tickCount        = 0;
int            g_orderCount       = 0;
ulong          g_lastReconnectMs  = 0;
int            g_hbReceived       = 0;

// Symbols - Elite 5 (no suffix)
string         g_symbols[];
int            g_symbolCount      = 5;

// Multi-Timeframe tracking
ENUM_TIMEFRAMES g_timeframes[];
string          g_tfNames[];
int             g_tfCount          = 4;

// Track last candle time per symbol to detect new candle
datetime       g_lastM5Time[];    // one per symbol

// ====================================================================

int OnInit()
{
   // Init symbols (no 'm' suffix - standard account)
   ArrayResize(g_symbols, 5);
   g_symbols[0] = "XAUUSD";
   g_symbols[1] = "US30";
   g_symbols[2] = "USTEC";
   g_symbols[3] = "BTCUSD";
   g_symbols[4] = "ETHUSD";

   // Init timeframes
   ArrayResize(g_timeframes, 4);
   g_timeframes[0] = PERIOD_M1;
   g_timeframes[1] = PERIOD_M5;
   g_timeframes[2] = PERIOD_M15;
   g_timeframes[3] = PERIOD_H1;

   ArrayResize(g_tfNames, 4);
   g_tfNames[0] = "M1";
   g_tfNames[1] = "M5";
   g_tfNames[2] = "M15";
   g_tfNames[3] = "H1";

   // Subscribe symbols
   for(int i = 0; i < g_symbolCount; i++)
      SymbolSelect(g_symbols[i], true);

   // Init last M5 time array
   ArrayResize(g_lastM5Time, g_symbolCount);
   ArrayInitialize(g_lastM5Time, 0);

   PrintFormat("RabitScal EA v3.1 started. Symbols: %d, Server: %s:%d",
               g_symbolCount, InpServerIP, InpPort);

   // Try initial connect
   ConnectToServer();

   return INIT_SUCCEEDED;
}

// ====================================================================

void OnDeinit(const int reason)
{
   if(g_socket != INVALID_HANDLE)
   {
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
   }
   g_connected = false;
   PrintFormat("EA deinitialized. Reason: %d", reason);
}

// ====================================================================

void OnTick()
{
   g_tickCount++;

   // 1. Ensure connection
   if(!g_connected || g_socket == INVALID_HANDLE)
   {
      ulong now = GetTickCount64();
      if((now - g_lastReconnectMs) >= (ulong)(InpReconnectDelay * 1000))
      {
         g_lastReconnectMs = now;
         ConnectToServer();
      }
      return;
   }

   // 2. Check for new M5 candle -> resend candles
   CheckNewCandleAndResend();

   // 3. Send tick data
   SendTickData();

   // 4. Read incoming (HEARTBEAT + ORDERS) - same socket
   ReadIncoming();
}

// ====================================================================
//  CONNECT
// ====================================================================

bool ConnectToServer()
{
   if(g_socket != INVALID_HANDLE)
   {
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
   }
   g_connected = false;

   g_socket = SocketCreate();
   if(g_socket == INVALID_HANDLE)
   {
      Print("ERR: SocketCreate failed: ", GetLastError());
      return false;
   }

   if(!SocketConnect(g_socket, InpServerIP, (uint)InpPort, (uint)InpConnectTimeout))
   {
      PrintFormat("ERR: SocketConnect %s:%d failed: %d", InpServerIP, InpPort, GetLastError());
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
      return false;
   }

   g_connected = true;
   g_lastHeartbeatMs = GetTickCount64();
   PrintFormat("Connected to %s:%d", InpServerIP, InpPort);

   // Send initial candles on connect
   SendInitialCandles();

   return true;
}

// ====================================================================
//  SEND TICK DATA
// ====================================================================

void SendTickData()
{
   string json = "{\"type\":\"TICK\",\"ts\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\",\"data\":[";

   for(int i = 0; i < g_symbolCount; i++)
   {
      double bid    = SymbolInfoDouble(g_symbols[i], SYMBOL_BID);
      double ask    = SymbolInfoDouble(g_symbols[i], SYMBOL_ASK);
      int spread    = (int)SymbolInfoInteger(g_symbols[i], SYMBOL_SPREAD);
      int digits    = (int)SymbolInfoInteger(g_symbols[i], SYMBOL_DIGITS);

      if(i > 0) json += ",";
      json += "{\"s\":\"" + g_symbols[i] + "\","
            + "\"b\":"  + DoubleToString(bid, digits) + ","
            + "\"a\":"  + DoubleToString(ask, digits) + ","
            + "\"sp\":" + IntegerToString(spread) + "}";
   }

   json += "]}";

   if(!SendMsg(g_socket, json))
   {
      Print("WARN: Tick send failed - disconnecting.");
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
      g_connected = false;
   }
}

// ====================================================================
//  SEND INITIAL CANDLES (all symbols x all TF)
// ====================================================================

void SendInitialCandles()
{
//  Format: {"type":"CANDLES","s":"XAUUSD","tf":"M5","c":300,
//           "d":[[t,o,h,l,c,v],...]}

   Print("Sending initial candles...");
   int sent = 0;
   for(int s = 0; s < g_symbolCount; s++)
   {
      for(int t = 0; t < g_tfCount; t++)
      {
         if(SendCandlesForSymbolTF(g_symbols[s], g_timeframes[t], g_tfNames[t]))
            sent++;
         else
            PrintFormat("WARN: Failed candles %s %s", g_symbols[s], g_tfNames[t]);
      }
   }
   PrintFormat("Initial candles sent: %d/%d", sent, g_symbolCount * g_tfCount);
}

// ====================================================================

bool SendCandlesForSymbolTF(string symbol, ENUM_TIMEFRAMES tf, string tfName)
{
   int count = InpCandleCount;
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(symbol, tf, 0, count, rates);
   if(copied <= 0)
   {
      PrintFormat("WARN: CopyRates %s %s returned %d", symbol, tfName, copied);
      return false;
   }

   // Build JSON: {"type":"CANDLES","s":"XAUUSD","tf":"M5","c":300,"sp":18,"d":[[t,o,h,l,c,v],...]}
   // "sp" = current market spread in POINTS at send-time (Phase 3 Spread Filter)
   int    spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   int    digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   string json = "{\"type\":\"CANDLES\","
               + "\"s\":\""  + symbol  + "\","
               + "\"tf\":\"" + tfName  + "\","
               + "\"c\":"    + IntegerToString(copied) + ","
               + "\"sp\":"   + IntegerToString(spread)  + ","
               + "\"d\":[";

   for(int i = copied - 1; i >= 0; i--)
   {
      if(i < copied - 1) json += ",";
      json += "[" + IntegerToString((int)rates[i].time)  + ","
                  + DoubleToString(rates[i].open,  digits) + ","
                  + DoubleToString(rates[i].high,  digits) + ","
                  + DoubleToString(rates[i].low,   digits) + ","
                  + DoubleToString(rates[i].close, digits) + ","
                  + IntegerToString((int)rates[i].tick_volume) + "]";
   }

   json += "]}";

   return SendMsg(g_socket, json);
}

// ====================================================================
//  CHECK NEW CANDLE - Resend candles when new M5 candle forms
// ====================================================================

void CheckNewCandleAndResend()
{
   for(int s = 0; s < g_symbolCount; s++)
   {
      datetime currentM5[];
      ArraySetAsSeries(currentM5, true);
      if(CopyTime(g_symbols[s], PERIOD_M5, 0, 1, currentM5) != 1)
         continue;

      if(g_lastM5Time[s] == 0)
      {
         // First check - just record
         g_lastM5Time[s] = currentM5[0];
         continue;
      }

      if(currentM5[0] > g_lastM5Time[s])
      {
         g_lastM5Time[s] = currentM5[0];
         // New M5 candle - resend all TF for this symbol
         for(int t = 0; t < g_tfCount; t++)
            SendCandlesForSymbolTF(g_symbols[s], g_timeframes[t], g_tfNames[t]);
      }
   }
}

// ====================================================================
//  READ INCOMING (HEARTBEAT + ORDERS) - SAME SOCKET
// ====================================================================

void ReadIncoming()
{
   if(g_socket == INVALID_HANDLE) return;

   // Check heartbeat timeout
   ulong nowMs = GetTickCount64();
   if((nowMs - g_lastHeartbeatMs) > (ulong)(InpHeartbeatMax * 1000))
   {
      Print("WARN: Heartbeat timeout - reconnecting.");
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
      g_connected = false;
      return;
   }

   // Try to read a message (non-blocking with short timeout)
   string msg = RecvMsg(g_socket, (uint)InpReadTimeout);
   if(msg == "")
   {
      // NO Disconnect() - one missed tick beat must NOT crash the bridge.
      // Print("WARN: Missed 1 tick beat - server may be busy, continuing...");
      return;
   }

   ProcessMessage(msg);
}

// ====================================================================

void ProcessMessage(string msg)
{
   // Parse type field
   int typeStart = StringFind(msg, "\"type\":\"");
   if(typeStart < 0) return;
   typeStart += 8;
   int typeEnd = StringFind(msg, "\"", typeStart);
   if(typeEnd < 0) return;
   string msgType = StringSubstr(msg, typeStart, typeEnd - typeStart);

   if(msgType == "HEARTBEAT")
   {
      g_lastHeartbeatMs = GetTickCount64();
      g_hbReceived++;
      return;
   }

   if(msgType == "ORDER")
   {
      g_orderCount++;
      ProcessOrder(msg);
      return;
   }

   if(msgType == "CLOSE_ALL")
   {
      EmergencyCloseAll();
      return;
   }

   PrintFormat("WARN: Unknown message type: %s", msgType);
}

// ====================================================================

void ProcessOrder(string msg)
{
   // Parse: {"type":"ORDER","id":"...","action":"BUY/SELL/CLOSE/CLOSE_ALL/MODIFY",
   //         "symbol":"...","lot":0.01,"sl":0,"tp":0,"ticket":0}

   string orderId = ParseStrField(msg, "id");
   string action  = ParseStrField(msg, "action");
   string symbol  = ParseStrField(msg, "symbol");
   double lot     = ParseDblField(msg, "lot");
   double sl      = ParseDblField(msg, "sl");
   double tp      = ParseDblField(msg, "tp");
   long   ticket  = (long)ParseDblField(msg, "ticket");

   PrintFormat("ORDER[%s] action=%s symbol=%s lot=%.2f sl=%.5f tp=%.5f ticket=%d",
               orderId, action, symbol, lot, sl, tp, (int)ticket);

   if(action == "BUY")
      ExecOpenPos(orderId, symbol, ORDER_TYPE_BUY, lot, sl, tp);
   else if(action == "SELL")
      ExecOpenPos(orderId, symbol, ORDER_TYPE_SELL, lot, sl, tp);
   else if(action == "CLOSE" && ticket > 0)
      ExecClosePos(orderId, ticket);
   else if(action == "CLOSE_ALL")
      ExecCloseAll(orderId, symbol);
   else if(action == "MODIFY" && ticket > 0)
      ExecModifyPos(orderId, ticket, sl, tp);
   else
      SendOrderResult(orderId, "REJECTED", 0, 0, 0, -1, "Unknown action: " + action);
}

// ====================================================================
//  JSON HELPERS
// ====================================================================

string ParseStrField(string json, string key)
{
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   pos += StringLen(search);
   int end = StringFind(json, "\"", pos);
   if(end < 0) return "";
   return StringSubstr(json, pos, end - pos);
}

double ParseDblField(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return 0;
   pos += StringLen(search);
   // Read until comma, } or end
   string val = "";
   for(int i = pos; i < StringLen(json); i++)
   {
      string c = StringSubstr(json, i, 1);
      if(c == "," || c == "}" || c == "]") break;
      val += c;
   }
   return StringToDouble(val);
}

// ====================================================================
//  SEND ORDER RESULT
// ====================================================================

void SendOrderResult(string orderId, string status, long ticket,
                     double price, double volume, int errCode, string errMsg)
{
   string json = "{\"type\":\"ORDER_RESULT\","
               + "\"id\":\""      + orderId              + "\","
               + "\"status\":\""  + status               + "\","
               + "\"ticket\":"    + IntegerToString(ticket) + ","
               + "\"price\":"     + DoubleToString(price, 5) + ","
               + "\"volume\":"    + DoubleToString(volume, 2) + ","
               + "\"err\":"       + IntegerToString(errCode) + ","
               + "\"msg\":\""     + errMsg               + "\"}";
   SendMsg(g_socket, json);
}

// ====================================================================
//  TRADE EXECUTION
// ====================================================================

void ExecOpenPos(string orderId, string symbol, ENUM_ORDER_TYPE orderType,
                 double volume, double sl, double tp)
{
   // === LENH TOI THUONG — Sep Vu 2026-03-15 ===
   // XAUUSD lot HARDCODED = 0.01. No exceptions.
   // Python risk guard also enforces this, but EA adds a second lock.
   if(symbol == "XAUUSD") volume = 0.01;

   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   request.action        = TRADE_ACTION_DEAL;
   request.symbol        = symbol;
   request.volume        = volume;
   request.type          = orderType;
   request.price         = (orderType == ORDER_TYPE_BUY)
                           ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                           : SymbolInfoDouble(symbol, SYMBOL_BID);
   request.sl            = sl;
   request.tp            = tp;
   request.magic         = InpMagicNumber;
   request.deviation     = InpMaxSlippage;
   request.type_filling  = ORDER_FILLING_IOC;

   bool success = OrderSend(request, result);
   if(success && result.retcode == TRADE_RETCODE_DONE)
      SendOrderResult(orderId, "FILLED", (long)result.deal, result.price, volume, 0, "");
   else
      SendOrderResult(orderId, "REJECTED", 0, 0, 0, (int)result.retcode, RetcodeStr(result.retcode));
}

void ExecClosePos(string orderId, long ticket)
{
   if(!PositionSelectByTicket((ulong)ticket))
   {
      SendOrderResult(orderId, "REJECTED", ticket, 0, 0, -1, "Not found");
      return;
   }
   string symbol  = PositionGetString(POSITION_SYMBOL);
   double volume  = PositionGetDouble(POSITION_VOLUME);
   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = volume;
   request.type      = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price     = (request.type == ORDER_TYPE_BUY)
                       ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(symbol, SYMBOL_BID);
   request.position  = ticket;
   request.deviation = InpMaxSlippage;
   request.type_filling = ORDER_FILLING_IOC;

   bool success = OrderSend(request, result);
   if(success && result.retcode == TRADE_RETCODE_DONE)
      SendOrderResult(orderId, "FILLED", ticket, result.price, volume, 0, "");
   else
      SendOrderResult(orderId, "REJECTED", ticket, 0, 0, (int)result.retcode, RetcodeStr(result.retcode));
}

void ExecCloseAll(string orderId, string symbol)
{
   int closed = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber && InpMagicNumber > 0) continue;
      ExecClosePos(orderId + "_" + IntegerToString(i), (long)t);
      closed++;
   }
   SendOrderResult(orderId, closed > 0 ? "FILLED" : "REJECTED", 0, 0, 0, 0,
                   "Closed " + IntegerToString(closed));
}

void ExecModifyPos(string orderId, long ticket, double sl, double tp)
{
   if(!PositionSelectByTicket((ulong)ticket))
   {
      SendOrderResult(orderId, "REJECTED", ticket, 0, 0, -1, "Not found");
      return;
   }
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   request.action   = TRADE_ACTION_SLTP;
   request.symbol   = PositionGetString(POSITION_SYMBOL);
   request.position = ticket;
   request.sl       = sl;
   request.tp       = tp;

   bool success = OrderSend(request, result);
   if(success && result.retcode == TRADE_RETCODE_DONE)
      SendOrderResult(orderId, "FILLED", ticket, 0, 0, 0, "");
   else
      SendOrderResult(orderId, "REJECTED", ticket, 0, 0, (int)result.retcode, RetcodeStr(result.retcode));
}

void EmergencyCloseAll()
{
   Print("*** EMERGENCY CLOSE ***");
   int closed = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber && InpMagicNumber > 0) continue;
      string symbol = PositionGetString(POSITION_SYMBOL);
      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      MqlTradeRequest request = {};
      MqlTradeResult  result  = {};
      request.action    = TRADE_ACTION_DEAL;
      request.symbol    = symbol;
      request.volume    = volume;
      request.type      = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      request.price     = (request.type == ORDER_TYPE_BUY)
                          ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                          : SymbolInfoDouble(symbol, SYMBOL_BID);
      request.position  = t;
      request.deviation = InpMaxSlippage * 3;
      request.type_filling = ORDER_FILLING_IOC;
      if(OrderSend(request, result)) closed++;
   }
   PrintFormat("Emergency: %d closed", closed);
}

// ====================================================================
//  NEWLINE-DELIMITED PROTOCOL (Sep Vu 2026-03-14)
// ====================================================================

string RecvMsg(int sock, uint timeoutMs)
{
   if(sock == INVALID_HANDLE) return "";

   // NEWLINE-DELIMITED PROTOCOL (Sep Vu 2026-03-14)
   // Read raw bytes, scan for '\n' delimiter to extract one JSON line.
   uchar buf[];
   ArrayResize(buf, 4096);
   int readBytes = SocketRead(sock, buf, 4096, timeoutMs);
   if(readBytes <= 0) return "";

   string raw = CharArrayToString(buf, 0, readBytes, CP_UTF8);

   // Find first complete line (ends with \n)
   int nlPos = StringFind(raw, "\n");
   if(nlPos < 0) return "";  // No complete message yet

   // Extract the first JSON line (before \n)
   string line = StringSubstr(raw, 0, nlPos);

   // Trim any stray \r (Windows CRLF safety)
   StringReplace(line, "\r", "");

   return line;
}

bool SendMsg(int sock, string message)
{
   if(sock == INVALID_HANDLE) return false;

   // NEWLINE-DELIMITED PROTOCOL (Sep Vu 2026-03-14)
   // No header! Just JSON + "\n" delimiter. Clean and simple.
   StringAdd(message, "\n");

   // Convert to UTF-8 bytes
   uchar payload[];
   int payloadLen = StringToCharArray(message, payload, 0, WHOLE_ARRAY, CP_UTF8);
   if(payloadLen > 0) payloadLen--;  // remove MQL5 null terminator from byte count

   // CHUNKED TRANSFER - send in 8KB chunks with retry on Error 5273
   int CHUNK_SIZE = 8192;
   int MAX_RETRIES = 100;
   int totalSent = 0;

   while(totalSent < payloadLen)
   {
      int remain = payloadLen - totalSent;
      int chunkSize = MathMin(remain, CHUNK_SIZE);

      uchar chunk[];
      ArrayResize(chunk, chunkSize);
      ArrayCopy(chunk, payload, 0, totalSent, chunkSize);

      int retries = 0;
      bool chunkSent = false;

      while(retries < MAX_RETRIES)
      {
         int bytesSent = SocketSend(sock, chunk, chunkSize);

         if(bytesSent == chunkSize)
         {
            totalSent += chunkSize;
            chunkSent = true;
            break;
         }
         else if(bytesSent > 0 && bytesSent < chunkSize)
         {
            totalSent += bytesSent;
            chunkSent = true;
            break;
         }
         else
         {
            retries++;
            Sleep(1);
         }
      }

      if(!chunkSent)
      {
         PrintFormat("ERR: SendMsg failed after %d retries at %d/%d bytes. Error: %d",
                     MAX_RETRIES, totalSent, payloadLen, GetLastError());
         return false;
      }
   }

   return true;
}

string RetcodeStr(uint retcode)
{
   if(retcode == TRADE_RETCODE_DONE)           return "Done";
   if(retcode == TRADE_RETCODE_PLACED)         return "Placed";
   if(retcode == TRADE_RETCODE_REQUOTE)        return "Requote";
   if(retcode == TRADE_RETCODE_REJECT)         return "Rejected";
   if(retcode == TRADE_RETCODE_ERROR)          return "Error";
   if(retcode == TRADE_RETCODE_INVALID)        return "Invalid";
   if(retcode == TRADE_RETCODE_INVALID_VOLUME) return "Bad volume";
   if(retcode == TRADE_RETCODE_INVALID_PRICE)  return "Bad price";
   if(retcode == TRADE_RETCODE_INVALID_STOPS)  return "Bad stops";
   if(retcode == TRADE_RETCODE_TRADE_DISABLED) return "Disabled";
   if(retcode == TRADE_RETCODE_MARKET_CLOSED)  return "Market closed";
   if(retcode == TRADE_RETCODE_NO_MONEY)       return "No money";
   return "Err" + IntegerToString(retcode);
}
//+------------------------------------------------------------------+
