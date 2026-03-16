//+------------------------------------------------------------------+
//| JAson.mqh — MQL5 JSON Parser v3.0 (RabitScal Edition)           |
//| Clean rewrite — all instance methods, zero static pointer issues|
//|                                                                  |
//| Usage:                                                            |
//|   CJAVal j;                                                       |
//|   j.Deserialize("{\"type\":\"BUY\",\"lot\":0.01}");               |
//|   string t = j.Get("type");        // "BUY"                      |
//|   double l = j.GetDbl("lot");      // 0.01                       |
//+------------------------------------------------------------------+
#ifndef JASON_MQH
#define JASON_MQH

//--- JSON value types
enum EnJAType { jtUNDEF, jtNULL, jtBOOL, jtINT, jtDBL, jtSTR, jtARR, jtOBJ };

//+------------------------------------------------------------------+
//| CJAVal                                                            |
//+------------------------------------------------------------------+
class CJAVal
{
public:
   string     m_key;
   EnJAType   m_type;
   bool       m_bv;
   long       m_iv;
   double     m_dv;
   string     m_sv;

   CJAVal    *m_e[];      // children
   int        m_size;

   //--- ctor / dtor
                CJAVal() { Clear(); }
               ~CJAVal() { FreeChildren(); }

   void        Clear()
   {
      m_key  = "";
      m_type = jtUNDEF;
      m_bv   = false;
      m_iv   = 0;
      m_dv   = 0.0;
      m_sv   = "";
      m_size = 0;
   }

   void        FreeChildren()
   {
      for(int i = 0; i < m_size; i++)
         if(m_e[i] != NULL)
            delete m_e[i];
      ArrayResize(m_e, 0);
      m_size = 0;
   }

   //--- getters
   string      ToStr()   { return m_sv;  }
   double      ToDbl()   { return m_dv;  }
   long        ToInt()   { return m_iv;  }
   bool        ToBool()  { return m_bv;  }
   int         Size()    { return m_size; }

   //--- named getters (safe, with defaults)
   string      Get(string k, string d = "")
   {
      CJAVal *c = FindKey(k);
      if(c != NULL) return c.m_sv;
      return d;
   }
   double      GetDbl(string k, double d = 0.0)
   {
      CJAVal *c = FindKey(k);
      if(c != NULL) return c.m_dv;
      return d;
   }
   long        GetInt(string k, long d = 0)
   {
      CJAVal *c = FindKey(k);
      if(c != NULL) return c.m_iv;
      return d;
   }
   bool        GetBool(string k, bool d = false)
   {
      CJAVal *c = FindKey(k);
      if(c != NULL) return c.m_bv;
      return d;
   }

   //--- child access
   CJAVal     *FindKey(string k)
   {
      for(int i = 0; i < m_size; i++)
         if(m_e[i] != NULL && m_e[i].m_key == k)
            return m_e[i];
      return NULL;
   }

   CJAVal     *At(int idx)
   {
      if(idx >= 0 && idx < m_size) return m_e[idx];
      return NULL;
   }

   void        AddChild(CJAVal *child)
   {
      ArrayResize(m_e, m_size + 1);
      m_e[m_size] = child;
      m_size++;
   }

   // ═══════════════════════════════════════════════════════════════
   //  DESERIALIZE
   // ═══════════════════════════════════════════════════════════════
   bool        Deserialize(string json)
   {
      FreeChildren();
      Clear();
      int pos = 0;
      return ParseValue(json, pos);
   }

   //--- parse any JSON value into 'this'
   bool        ParseValue(string src, int &pos)
   {
      SkipWS(src, pos);
      int len = StringLen(src);
      if(pos >= len) return false;

      ushort ch = StringGetCharacter(src, pos);

      if(ch == '{')  return ParseObj(src, pos);
      if(ch == '[')  return ParseArr(src, pos);
      if(ch == '"')  return ParseStr(src, pos);
      if(ch == 't' || ch == 'f') return ParseBoolVal(src, pos);
      if(ch == 'n')  return ParseNullVal(src, pos);
      if(ch == '-' || (ch >= '0' && ch <= '9')) return ParseNum(src, pos);

      return false;
   }

   // ─── Object ─────────────────────────────────────────────────
   bool        ParseObj(string src, int &pos)
   {
      pos++;  // '{'
      m_type = jtOBJ;
      SkipWS(src, pos);

      if(CharAt(src, pos) == '}') { pos++; return true; }

      for(;;)
      {
         SkipWS(src, pos);
         if(CharAt(src, pos) != '"') return false;

         // parse key
         string key = "";
         if(!ReadString(src, pos, key)) return false;

         SkipWS(src, pos);
         if(CharAt(src, pos) != ':') return false;
         pos++;

         // parse value into new child
         CJAVal *child = new CJAVal();
         child.m_key = key;
         if(!child.ParseValue(src, pos))
         {
            delete child;
            return false;
         }
         AddChild(child);

         SkipWS(src, pos);
         ushort nx = CharAt(src, pos);
         if(nx == '}') { pos++; return true; }
         if(nx == ',') { pos++; continue; }
         return false;
      }
      return false;
   }

   // ─── Array ──────────────────────────────────────────────────
   bool        ParseArr(string src, int &pos)
   {
      pos++;  // '['
      m_type = jtARR;
      SkipWS(src, pos);

      if(CharAt(src, pos) == ']') { pos++; return true; }

      int idx = 0;
      for(;;)
      {
         CJAVal *child = new CJAVal();
         child.m_key = IntegerToString(idx);
         idx++;
         if(!child.ParseValue(src, pos))
         {
            delete child;
            return false;
         }
         AddChild(child);

         SkipWS(src, pos);
         ushort nx = CharAt(src, pos);
         if(nx == ']') { pos++; return true; }
         if(nx == ',') { pos++; continue; }
         return false;
      }
      return false;
   }

   // ─── String ─────────────────────────────────────────────────
   bool        ParseStr(string src, int &pos)
   {
      string val = "";
      if(!ReadString(src, pos, val)) return false;
      m_type = jtSTR;
      m_sv   = val;
      return true;
   }

   // ─── Bool ───────────────────────────────────────────────────
   bool        ParseBoolVal(string src, int &pos)
   {
      if(StringSubstr(src, pos, 4) == "true")
      {
         m_type = jtBOOL;
         m_bv = true;  m_iv = 1;  m_dv = 1.0;  m_sv = "true";
         pos += 4;
         return true;
      }
      if(StringSubstr(src, pos, 5) == "false")
      {
         m_type = jtBOOL;
         m_bv = false;  m_iv = 0;  m_dv = 0.0;  m_sv = "false";
         pos += 5;
         return true;
      }
      return false;
   }

   // ─── Null ───────────────────────────────────────────────────
   bool        ParseNullVal(string src, int &pos)
   {
      if(StringSubstr(src, pos, 4) == "null")
      {
         m_type = jtNULL;
         m_sv = "";
         pos += 4;
         return true;
      }
      return false;
   }

   // ─── Number ─────────────────────────────────────────────────
   bool        ParseNum(string src, int &pos)
   {
      int start = pos;
      bool isFloat = false;
      int len = StringLen(src);

      if(CharAt(src, pos) == '-') pos++;

      while(pos < len)
      {
         ushort c = CharAt(src, pos);
         if(c >= '0' && c <= '9') { pos++; continue; }
         if(c == '.' || c == 'e' || c == 'E') { isFloat = true; pos++; continue; }
         if(c == '+' || c == '-')
         {
            // only valid after e/E
            if(pos > start)
            {
               ushort prev = CharAt(src, pos - 1);
               if(prev == 'e' || prev == 'E') { pos++; continue; }
            }
         }
         break;
      }

      string ns = StringSubstr(src, start, pos - start);
      if(isFloat)
      {
         m_type = jtDBL;
         m_dv = StringToDouble(ns);
         m_iv = (long)m_dv;
      }
      else
      {
         m_type = jtINT;
         m_iv = StringToInteger(ns);
         m_dv = (double)m_iv;
      }
      m_sv = ns;
      return true;
   }

   // ═══════════════════════════════════════════════════════════════
   //  SERIALIZE
   // ═══════════════════════════════════════════════════════════════
   string      Serialize()
   {
      return SerNode();
   }

   string      SerNode()
   {
      switch(m_type)
      {
         case jtNULL: return "null";
         case jtBOOL: return (m_bv ? "true" : "false");
         case jtINT:  return IntegerToString(m_iv);
         case jtDBL:  return DoubleToString(m_dv, 8);
         case jtSTR:  return "\"" + EscStr(m_sv) + "\"";
         case jtARR:
         {
            string a = "[";
            for(int i = 0; i < m_size; i++)
            {
               if(i > 0) a += ",";
               if(m_e[i] != NULL) a += m_e[i].SerNode();
               else               a += "null";
            }
            return a + "]";
         }
         case jtOBJ:
         {
            string o = "{";
            for(int i = 0; i < m_size; i++)
            {
               if(i > 0) o += ",";
               if(m_e[i] != NULL)
                  o += "\"" + EscStr(m_e[i].m_key) + "\":" + m_e[i].SerNode();
            }
            return o + "}";
         }
      }
      return "null";
   }

   // ═══════════════════════════════════════════════════════════════
   //  HELPERS
   // ═══════════════════════════════════════════════════════════════
   ushort      CharAt(string s, int p)
   {
      if(p < 0 || p >= StringLen(s)) return 0;
      return StringGetCharacter(s, p);
   }

   void        SkipWS(string s, int &p)
   {
      int len = StringLen(s);
      while(p < len)
      {
         ushort c = StringGetCharacter(s, p);
         if(c == ' ' || c == '\t' || c == '\r' || c == '\n')
            p++;
         else
            break;
      }
   }

   //--- read a quoted string, advance pos past closing quote
   bool        ReadString(string src, int &pos, string &out)
   {
      if(CharAt(src, pos) != '"') return false;
      pos++;  // skip opening '"'
      out = "";
      int len = StringLen(src);

      while(pos < len)
      {
         ushort c = StringGetCharacter(src, pos);
         pos++;

         if(c == '"') return true;  // end of string

         if(c == '\\' && pos < len)
         {
            ushort e = StringGetCharacter(src, pos);
            pos++;
            if(e == '"')       out += "\"";
            else if(e == '\\') out += "\\";
            else if(e == '/')  out += "/";
            else if(e == 'n')  out += "\n";
            else if(e == 'r')  out += "\r";
            else if(e == 't')  out += "\t";
            else               out += ShortToString(e);
         }
         else
         {
            out += ShortToString(c);
         }
      }
      return false;  // unterminated
   }

   string      EscStr(string s)
   {
      string r = "";
      int len = StringLen(s);
      for(int i = 0; i < len; i++)
      {
         ushort c = StringGetCharacter(s, i);
         if(c == '"')        r += "\\\"";
         else if(c == '\\')  r += "\\\\";
         else if(c == '\n')  r += "\\n";
         else if(c == '\r')  r += "\\r";
         else if(c == '\t')  r += "\\t";
         else                r += ShortToString(c);
      }
      return r;
   }
};

#endif // JASON_MQH
