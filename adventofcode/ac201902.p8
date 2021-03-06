pico-8 cartridge // http://www.pico-8.com
version 29
__lua__
-- advent of code, 2019 day 2
-- by sestrenexsis

-- https://creativecommons.org/licenses/by-nc-sa/4.0/

_code={
	"1",
	"0",
	"0",
	"3",
	"1",
	"1",
	"2",
	"3",
	"1",
	"3",
	"4",
	"3",
	"1",
	"5",
	"0",
	"3",
	"2",
	"13",
	"1",
	"19",
	"1",
	"6",
	"19",
	"23",
	"2",
	"6",
	"23",
	"27",
	"1",
	"5",
	"27",
	"31",
	"2",
	"31",
	"9",
	"35",
	"1",
	"35",
	"5",
	"39",
	"1",
	"39",
	"5",
	"43",
	"1",
	"43",
	"10",
	"47",
	"2",
	"6",
	"47",
	"51",
	"1",
	"51",
	"5",
	"55",
	"2",
	"55",
	"6",
	"59",
	"1",
	"5",
	"59",
	"63",
	"2",
	"63",
	"6",
	"67",
	"1",
	"5",
	"67",
	"71",
	"1",
	"71",
	"6",
	"75",
	"2",
	"75",
	"10",
	"79",
	"1",
	"79",
	"5",
	"83",
	"2",
	"83",
	"6",
	"87",
	"1",
	"87",
	"5",
	"91",
	"2",
	"9",
	"91",
	"95",
	"1",
	"95",
	"6",
	"99",
	"2",
	"9",
	"99",
	"103",
	"2",
	"9",
	"103",
	"107",
	"1",
	"5",
	"107",
	"111",
	"1",
	"111",
	"5",
	"115",
	"1",
	"115",
	"13",
	"119",
	"1",
	"13",
	"119",
	"123",
	"2",
	"6",
	"123",
	"127",
	"1",
	"5",
	"127",
	"131",
	"1",
	"9",
	"131",
	"135",
	"1",
	"135",
	"9",
	"139",
	"2",
	"139",
	"6",
	"143",
	"1",
	"143",
	"5",
	"147",
	"2",
	"147",
	"6",
	"151",
	"1",
	"5",
	"151",
	"155",
	"2",
	"6",
	"155",
	"159",
	"1",
	"159",
	"2",
	"163",
	"1",
	"9",
	"163",
	"0",
	"99",
	"2",
	"0",
	"14",
	"0"
	}

_code2={
	"1",
	"5",
	"6",
	"7",
	"99",
	"2",
	"9",
	"0",
	"0",
}
-->8
-- decimal floating point library
-- by jwinslow23
-- https://www.lexaloffle.com/bbs/?tid=39319
-- https://creativecommons.org/licenses/by-nc-sa/4.0/

function clone(t)
	local n={}
	for k,v in pairs(t) do
		n[k]=v
	end
	return n
end

function str_cmp(s,t)
	if (#s!=#t) return #s-#t
	for i=1,#s do
		local d=subc(s,i)-subc(t,i)
		if (d!=0) return d
	end
	return 0
end

function str_here(s,t,i,a)
	i=i or 1
	return sub(t,i,a and #t or i+#s-1)==s
end

function strip_leading(s)
	local zero_split,i=split(s,"0"),1
	if (#zero_split>#s) return "0"
	while zero_split[i]=="" do
		i+=1
	end
	return sub(s,i)
end

function subc(s,i)
	return sub(s,i,i)
end

--invalid operation
function __invalid_operation(cond)
	assert(not cond,"invalid operation")
end

--new decimal float
function df_new(num,prec)
	num=num or {"+","0","qnan"}

	if type(num)!="table" then
		num=df_parse(tostr(num))
	end

	add(num,prec or 32)
	return df_limit(num)
end
function df_val(abstract,...)
	local vals,prec={...},0x7fff.ffff
	for v in all(vals) do
		prec=min(prec,v[4])
	end
	add(abstract,prec)
	return abstract
end
function df_single(num)
	return df_new(num,32)
end
function df_double(num)
	return df_new(num,64)
end
function df_quad(num)
	return df_new(num,128)
end

--round/overflow/underflow
function df_limit(val)
	return df_underflow(df_overflow(df_round(val)))
end

--check if finite
function df_is_finite(val)
	return type(val[3])=="number"
end

--check if nan
function df_is_nan(val)
	return str_here("nan",val[3],2)
end

--check if snan
function df_is_snan(val)
	return val[3]=="snan"
end

--check sign
function df_sign(val,zero)
	return zero and val[2]=="0" and val[3]!="inf" and 0 or val[1]=="-" and -1 or 1
end

--check payload
function df_payload(val)
	if (df_is_nan(val)) return val[2]
end

--convert to string
function df_tostr(val)
	local str=df_sign(val)<0 and "-" or ""

	if df_is_finite(val) then
		local coefficient,exponent=val[2],val[3]
		local adj_exp=exponent+#coefficient-1
	
		--character form
		if exponent<=0 and adj_exp>=-6 then
			if exponent==0 then
				str..=coefficient
			else
				while #coefficient<-exponent do
					coefficient="0"..coefficient
				end
				str..=sub("0"..coefficient,#coefficient<=-exponent and 1 or 2,exponent-1).."."..sub(coefficient,exponent)
			end
		--exponential notation
		else
			if (#coefficient>1) coefficient=subc(coefficient,1).."."..sub(coefficient,2)
			str..=coefficient.."e"..(exponent>0 and "+" or "-")..abs(adj_exp)
		end
	else
		str..=df_is_snan(val) and "snan" or df_is_nan(val) and "nan" or "infinity"
	end

	return str
end

--rounding values
function df_round(val,p)
	p=p or val[4]\32*9-2

	if df_is_finite(val) then
		if (p==0) return df_val({"+","0",0},val)
		local val_c=strip_leading(val[2])
		if (#val_c<=p) val[2]=val_c return val
	
		local res_s,res_c,res_e,carry,i=val[1],"",val[3]+#val_c-p,subc(val_c,p+1)>="5" and 1 or 0,p
		repeat
			carry+="0"..sub(val_c,max(i-3),max(i))
			res_c=sub("000"..carry%10000,-4)..res_c
			carry\=10000
			i-=4
		until i<1 and carry<=0
		res_c=strip_leading(res_c)
	
		if #res_c>p then
			res_c=sub(res_c,1,p)
			res_e+=1
		end
	
		return df_val({res_s,res_c,res_e},val)
	end

	local v=clone(val)
	v[2]=p>1 and df_is_nan(val) and sub(v[2],1-p) or "0"
	return v
end

function df_overflow(val)
	if (not df_is_finite(val)) return val
	if (val[3]+#val[2]-2<=(3*2^(val[4]\32*2+4)-1)\2) return val
	return df_val({val[1],"0","inf"},val)
end

function df_underflow(val)
	if (not df_is_finite(val)) return val
	local v=clone(val)
	local e_min=-((3*2^(v[4]\32*2+4)-1)\2)
	if (v[3]+#v[2]-1>e_min) return val
	while v[3]<e_min-v[4]\32*9+3 do
		v[2]=sub(v[2],1,-2)
		v[3]+=1
	end
	if (v[2]=="") v[2]="0"
	return v
end

--compare values
function df_compare(val1,val2)
	local v1,v2=clone(val1),clone(val2)

	--handle nans
	if df_is_snan(val1) then
		v1[3]="qnan"
		return v1
	end
	if df_is_snan(val2) then
		v2[3]="qnan"
		return v2
	end
	if (df_is_nan(v1)) return v1
	if (df_is_nan(v2)) return v2

	if df_sign(v1,true)!=df_sign(v2,true)
	then
		v1_s,v2_s=df_sign(v1,true),df_sign(v2,true)
		v1,v2=df_val({subc("-++",v1_s+2),tostr(v1_s),0},v1),df_val({subc("-++",v2_s+2),tostr(v2_s),0},v2)
	end

	return df_sign(df_subtract(v1,v2),true)
end

--div and mod simultaneously
function df_divmod(val1,val2)
	local i,r=df_divide(val1,val2,true)
	assert(#i[2]+i[3]<=i[4]\32*9-2,"division impossible")
	while i[3]>0 do
		i[2]..="0"
		i[3]-=1
	end
	return df_limit(i),r
end

--x\y
function df_div(val1,val2)
	local val=df_divmod(val1,val2)
	return val
end

--x+y
function df_add(val1,val2)
	--handle nans
	__invalid_operation(df_is_snan(val1) or df_is_snan(val2))
	if (df_is_nan(val1)) return clone(val1)
	if (df_is_nan(val2)) return clone(val2)

	local val1_s,val2_s,val1_c,val2_c,val1_e,val2_e=val1[1],val2[1],val1[2],val2[2],val1[3],val2[3]

	--handle infinities
	if val1_e=="inf" then
		__invalid_operation(val2_e=="inf" and val1_s!=val2_s)
		return df_val({val1_s,"0","inf"},val1)
	elseif val2_e=="inf" then
		return df_val({val2_s,"0","inf"},val2)
	end

	for i=1,abs(val1_e-val2_e) do
		if val1_e>val2_e then
			val1_c..="0"
		else
			val2_c..="0"
		end
	end

	local val1_smaller,res_e,res_c,carry,i=str_cmp(val1_c,val2_c)<0,min(val1_e,val2_e),"",0,1
	repeat
		local val1_d,val2_d="0"..sub(val1_c,-i-2,-i),"0"..sub(val2_c,-i-2,-i)
		if val1_s!=val2_s then
			if val1_smaller then
				carry+=val2_d-val1_d
			else
				carry+=val1_d-val2_d
			end
		else
			carry+=val1_d+val2_d
		end
		res_c=sub("00"..carry%1000,-3)..res_c
		carry\=1000
		i+=3
	until i>max(#val1_c,#val2_c) and carry<=0

	local res_s,res_zero="+",true
	for i=1,#res_c do
		if (subc(res_c,i)!="0") res_zero=false break
	end
	if res_zero then
		if (val1_s=="-" and val2_s=="-") res_s="-"
	else
		res_s=val1_smaller and val2_s or val1_s
	end

	return df_limit(df_val({res_s,res_c,res_e},val1,val2))
end

--x-y
function df_subtract(val1,val2)
	local v2=clone(val2)
	v2[1]=v2[1]=="+" and "-" or "+"
	return df_add(val1,v2)
end

--x*y
function df_multiply(val1,val2)
	--handle nans
	__invalid_operation(df_is_snan(val1) or df_is_snan(val2))
	if (df_is_nan(val1)) return clone(val1)
	if (df_is_nan(val2)) return clone(val2)

	local res_s,val1_c,val2_c,val1_e,val2_e=val1[1]==val2[1] and "+" or "-",val1[2],val2[2],val1[3],val2[3]

	--handle infinities
	if val1_e=="inf" then
		__invalid_operation(df_sign(val2,true)==0)
		return df_val({res_s,"0","inf"},val2)
	elseif val2_e=="inf" then
		__invalid_operation(df_sign(val1,true)==0)
		return df_val({res_s,"0","inf"},val1)
	end

	local res_c,res_e,digits,carry="",val1_e+val2_e,{},0
	for j=1,#val2_c,2 do
		local i=1
		repeat
			carry+=(digits[i\2+j\2+1] or 0)+("0"..sub(val1_c,-i-1,-i))*("0"..sub(val2_c,-j-1,-j))
			digits[i\2+j\2+1]=carry%100
			carry\=100
			i+=2
		until i>#val1_c and carry<=0
	end

	for i=1,#digits do
		res_c=sub("0"..digits[i],-2)..res_c
	end

	return df_limit(df_val({res_s,res_c,res_e},val1,val2))
end

--x/y
function df_divide(val1,val2,int)
	--handle nans
	__invalid_operation(df_is_snan(val1) or df_is_snan(val2))
	if (df_is_nan(val1)) return clone(val1)
	if (df_is_nan(val2)) return clone(val2)

	local p1,p2,val1_c,val2_c,val1_e,val2_e=val1[4]\32*9-2,val2[4]\32*9-2,val1[2],val2[2],val1[3],val2[3]
	local res_s=val1[1]==val2[1] and "+" or "-"

	--handle infinities
	if val1_e=="inf" then
		__invalid_operation(val2_e=="inf")
		return df_val({res_s,"0","inf"},val2)
	elseif val2_e=="inf" then
		return df_val({res_s,"0",0},val1)
	end

	local adjust,res_c,res_e,new_digit=0,"",val1_e-val2_e,0

	if df_sign(val2,true)==0 then
		assert(df_sign(val1,true)!=0,"division undefined")
		return df_val({res_s,"0","inf"},val1,val2)
	end

	if df_sign(val1,true)==0 or int and df_lt(val1,val2) then
		res_c="0"
	else
		while str_cmp(val1_c,val2_c)<0 do
			val1_c..="0"
			val1_e-=1
			adjust+=1
		end
		while str_cmp(val1_c,val2_c.."0")>=0 do
			val2_c..="0"
			adjust-=1
		end
	
		res_c=""
	
		local lut={[0]="0",val2_c}
		while true do
			while str_cmp(lut[new_digit],val1_c)<=0 do
				new_digit+=1
			
				if not lut[new_digit] then
					local mult_new,carry,i="",0,1
					repeat
						carry+=("0"..sub(lut[new_digit-1],-i-3,-i))+("0"..sub(val2_c,-i-3,-i))
						mult_new=sub("000"..carry%10000,-4)..mult_new
						carry\=10000
						i+=4
					until i>max(#lut[new_digit-1],#val1_c) and carry<=0
					lut[new_digit]=strip_leading(mult_new)
				end
			end
		
			new_digit-=1
		
			local val1_c_new,carry,i="",0,1
			repeat
				carry+=("0"..sub(val1_c,-i-3,-i))-("0"..sub(lut[new_digit],-i-3,-i))
				val1_c_new=sub("000"..carry%10000,-4)..val1_c_new
				carry\=10000
				i+=4
			until i>max(#val1_c,#lut[new_digit]) and carry<=0
			val1_c=strip_leading(val1_c_new)
		
			res_c..=new_digit
		
			if val1_c=="0" and adjust>=0
			or #res_c>min(p1,p2)
			or int and res_e<=adjust
			then
				break
			end
		
			new_digit=0
			val1_c..="0"
			val1_e-=1
			adjust+=1
		end
	end

	local res=df_limit(df_val({res_s,res_c,res_e-adjust},val1,val2))

	if int then
		for i=2,#res_c do
			if (subc(val1_c,-1)!="0") break
			val1_c=sub(val1_c,1,-2)
			val1_e+=1
		end
		return res,df_limit(df_val({val1[1],val1_c,val1_e},val1,val2))
	end

	return res
end

--x%y
function df_mod(val1,val2)
	local _,val=df_divmod(val1,val2)
	return val
end

--x==y
function df_eq(val1,val2)
	return df_compare(val1,val2)==0
end

--x<y
function df_lt(val1,val2)
	return df_compare(val1,val2)<0
end

--x<=y
function df_le(val1,val2)
	return df_compare(val1,val2)<=0
end

-- -x
function df_unm(val)
	return df_subtract(df_val({"+","0",0},val),val)
end

--less common functions

--abs(x)
--[[
function df_abs(val)
	if (df_sign(val)<0) return df_unm(val)
	return val
end
--]]

--ceil(x)
--[[
function df_ceil(val)
	local v=df_ipart(val)
	if (df_lt(v,val)) v=df_add(v,df_val({"+","1",0},v))
	return v
end
--]]

--flr(x)
--[[
function df_flr(val)
	local v=val:ipart()
	if (df_lt(val,v)) v=df_add(v,df_val({"-","1",0},v))
	return v
end
--]]

--fractional part
---[[
function df_fpart(val)
	local v=clone(val)

	--handle snan
	__invalid_operation(df_is_snan(v))
	--handle nan/infinity
	if (df_is_nan(v) or not df_is_finite(v)) return v

	v[2]=v[3]<0 and v[3]<=#v[2] and sub(v[2],v[3]) or "0"
	return df_limit(v)
end
--]]

--integer part
---[[
function df_ipart(val)
	local v=clone(val)

	--handle snan
	__invalid_operation(df_is_snan(v))
	--handle nan/infinity
	if (df_is_nan(v) or not df_is_finite(v)) return v

	if v[3]<0 then
		v[2],v[3]=sub(v[2],1,v[3]-1),0
	end
	if (v[2]=="") v[2]="0"

	return df_limit(v)
end
--]]

--max(x,y)
--[[
function df_max(val1,val2)
	local v1,v2=clone(val1),clone(val2)

	--handle nans
	__invalid_operation(df_is_snan(v1) or df_is_snan(v2))
	if (not df_is_nan(v1) and df_is_nan(v2)) return v1
	if (not df_is_nan(v2) and df_is_nan(v1)) return v2
	if (df_is_nan(v1) and df_is_nan(v2)) return v1

	local cmp=df_compare(v1,v2)
	if cmp==0 then
		if (df_sign(v1)!=df_sign(v2)) return df_sign(v1)>0 and v1 or v2
		if (v1[3]==v2[3]) return v1
		if (df_sign(v1)>0) return v1[3]>v2[3] and v1 or v2
		return v1[3]<v2[3] and v1 or v2
	end

	return cmp==1 and v1 or v2
end
--]]

--min(x,y)
--[[
function df_min(val1,val2)
	local v1,v2=clone(val1),clone(val2)

	--handle nans
	__invalid_operation(df_is_snan(v1) or df_is_snan(v2))
	if (not df_is_nan(v1) and df_is_nan(v2)) return v1
	if (not df_is_nan(v2) and df_is_nan(v1)) return v2
	if (df_is_nan(v1) and df_is_nan(v2)) return v1

	local cmp=df_compare(v1,v2)
	if cmp==0 then
		if (df_sign(v1)!=df_sign(v2)) return df_sign(v1)<0 and v1 or v2
		if (v1[3]==v2[3]) return v1
		if (df_sign(v1)>0) return v1[3]<v2[3] and v1 or v2
		return v1[3]>v2[3] and v1 or v2
	end

	return cmp==-1 and v1 or v2
end
--]]

--remove trailing zeros
--[[
function df_reduce(val)
	local v=df_add(df_val({"+","0",0},val),val)

	--handle nan/infinity
	--(no need to handle snan)
	if (df_is_nan(v) or not df_is_finite(v)) return v

	while subc(v[2],-1)=="0" do
		v[2]=sub(v[2],1,-2)
		v[3]+=1
	end
	if (v[2]=="") v[2]="0"

	return df_limit(v)
end
]]

--conversion to/from bytes

--[[

function bin(n,l)
	local s=""
	for i=1,l or 8 do
		s=(n&1)..s
		n>>=1
	end
	return s
end

--convert to bytes
function df_tobytes(val)
	local p,ecbits=val[4]\32*9-2,val[4]\32*2+4
	local bias,coefficient=p+(3*2^ecbits-1)\2-1,val[2]
	while #coefficient<p do
		coefficient="0"..coefficient
	end
	local enc_exp,bits=type(val[3])=="number" and val[3]+bias or 0,val[1]=="-" and 1 or 0

	--combination field
	if df_is_finite(val) then
		local c_msd,e_msbs=subc(coefficient,1),enc_exp>>ecbits&3
		bits..=(c_msd<"8" and bin(e_msbs,2)..bin(c_msd,3) or "11"..bin(e_msbs,2)..(c_msd&1))
	elseif df_is_nan(val) then
		bits..="11111"
	else
		bits..="11110"
	end

	--exponent continuation
	local ec,bytes=bin(enc_exp,ecbits),{}
	if df_is_snan(val) then
		ec="1"..sub(ec,2)
	elseif df_is_nan(val) then
		ec="0"..sub(ec,2)
	end
	bits..=ec

	--coefficient continuation
	for i=2,#coefficient,3 do
		local n="0x"..sub(coefficient,i,i+2)%1000
		for c in all(split(split"bcdfgh0jkl,bcdfgh100l,bcdjkh101l,bcd10h111l,jkdfgh110l,fgd01h111l,jkd00h111l,00d11h111l"[(n>>9&4)+(n>>6&2)+(n>>3&1)+1],"")) do
			bits..=type(c)=="number" and c or n>>(108-ord(c))&1
		end
	end

	for i=1,#bits,8 do
		add(bytes,tonum("0b"..sub(bits,i,i+7)))
	end
	return bytes
end

--convert bytes to number
function df_from(bytes)
	local bits=""
	for i in all(bytes) do
		bits..=bin(i)
	end

	local p,ecbits=#bits\32*9-2,#bits\32*2+4
	local e_min=-((3*2^ecbits-1)\2)
	local bias,abstract=p-e_min-1,{subc(bits,1)=="1" and "-" or "+",""}

	local cf=sub(bits,2,7)
	if str_here("11111",cf) then
		add(abstract,split"qnan,snan"[subc(cf,6)+1])
	elseif str_here("11110",cf) then
		add(abstract,"inf")
	elseif str_here("11",cf) then
		abstract[2]=tostr(tonum("0b100"..subc(cf,5)))
		add(abstract,tonum("0b"..sub(cf,3,4)..sub(bits,7,6+ecbits))-bias)
	else
		abstract[2]=tostr(tonum("0b"..sub(cf,3,5)))
		add(abstract,tonum("0b"..sub(cf,1,2)..sub(bits,7,6+ecbits))-bias)
	end

	for i=7+ecbits,#bits,10 do
		local n,b="0b"..sub(bits,i,i+9),0
		for c in all(split(n&8>0 and (n&6==6 and split"100r100u0pqy,100r0pqu100y,0pqr100u100y,100r100u100y"[(n>>5&3)+1] or split"0pqr0stu100y,0pqr100u0sty,100r0stu0pqy"[(n>>1&3)+1]) or "0pqr0stu0wxy","")) do
			b=b*2+(type(c)=="number" and c or n>>(121-ord(c))&1)
		end
		abstract[2]..=sub("00"..sub(tostr(b,true),4,6),-3)
	end
	abstract[2]=strip_leading(abstract[2])

	return df_new(abstract,#bits)
end

--]]

--conversion from string

---[[

function extract_num(str,int,exp,exists,unsigned)
	local sign,exp_part="+"
	if str_here("-",str) then
		sign,str="-",sub(str,2)
		if (unsigned) return {}
	elseif str_here("+",str) then
		str=sub(str,2)
		if (unsigned) return {}
	end

	if exp then
		local exp_split=split(str,"e",false)
		if #exp_split==2 then
			local pwr=extract_num(exp_split[2],true,false,true)
			if (#pwr>0) exp_part,str=pwr[1],exp_split[1]
		end
	end

	if (str==".") return {}
	if (str=="") return exists and {} or {""}

	if int then
		for i=1,#str do
			if (not tonum(subc(str,i))) return {}
		end
		if (exp) return {sign..str,"",exp_part}
		return {sign..str}
	end

	local dot_split=split(str,".",false)
	if (#dot_split>2) return {}
	return {sign..dot_split[1],dot_split[2] or "",exp_part}
end

--parse string into abstract
function df_parse(str)
	local invalid,sign,exponent,i={"+","0","qnan"},"+",0,1
	--process sign, if any
	if str_here("+",str) then
		sign="+"
		i+=1
	elseif str_here("-",str) then
		sign="-"
		i+=1
	end

	--process infinity
	if str_here("inf",str,i,true)
	or str_here("infinity",str,i,true)
	then
		return {sign,"0","inf"}
	end

	--process nans
	for j=1,2 do
		if str_here(split"nan,snan"[j],str,i) then
			i+=j+2
			local num=extract_num(sub(str,i),true,false,false,true)
			if (#num==0) return invalid
			return {sign,num[1]=="" and "0" or sub(num[1],2),split"qnan,snan"[j]}
		end
	end

	local num=extract_num(sub(str,i),false,true)
	if (#num<2) return invalid

	return {sign,strip_leading(sub(num[1],2)..num[2]),(num[3] or 0)-#num[2]}
end

--]]
-->8
-- virtual machine
fadd=df_add
fmul=df_multiply
fstr=df_tostr

intcode={}

function intcode:new(
	t, --table of str numbers
	f  --function for numbers
	)
	local obj={
		pc=0,      -- program counter
		cy=0,      -- cycles
		hlt=false, -- halted?
		ram={},    -- memory
		cmd="",    -- last command
	}
	if f==nil then
		f=df_double
	end
	local idx=0
	for s in all(t) do
		obj.ram[idx]=f(s)
		idx+=1
	end
	return setmetatable(
		obj,{__index=self}
	)
end

function intcode:addr(f)
	local idx=tonum(fstr(f))
	local res={
		k=idx,
		v=self.ram[idx],
		}
	return res
end

function intcode:tick()
	if self.hlt then
		return false
	end
	local n0=self.ram[self.pc]
	local n1=self.ram[self.pc+1]
	local n2=self.ram[self.pc+2]
	local n3=self.ram[self.pc+3]
	local op=tonum(fstr(n0))
	if op==1 then
		local a=self:addr(n1)
		local b=self:addr(n2)
		local c=self:addr(n3)
		self.ram[c.k]=fadd(a.v,b.v)
		self.pc+=4
		self.cmd="add "..a.k.." "..b.k.." "..c.k
	elseif op==2 then
		local a=self:addr(n1)
		local b=self:addr(n2)
		local c=self:addr(n3)
		self.ram[c.k]=fmul(a.v,b.v)
		self.pc+=4
		self.cmd="mul "..a.k.." "..b.k.." "..c.k
	else --op==99
		self.cmd="hlt"
		self.hlt=true
	end
	self.cy+=1
	return true
end

function intcode:run()
	while not self.hlt do
		self:tick()
	end
end

function intcode:draw(
	x,
	y
	)
	print(self.pc,x,y)
	print(self.cy,x,y+6)
	print(fstr(self.ram[0]),x,y+18)
end
-->8
-- main

function _init()
	_f=df_double
	-- part one
	_vm1=intcode:new(_code,_f)
	_vm1.ram[1]=_f("12")
	_vm1.ram[2]=_f("2")
	-- part two
	_iter=2300
	_vm2=intcode:new(_code,_f)
	_vm2.ram[1]=_f("0")
	_vm2.ram[2]=_f("0")
	-- command histories
	_cmds={{},{}}
	_target=_f("19690720")
end

function _update()
	if _vm1:tick() then
		add(_cmds[1],_vm1.cmd)
	end
	local halt=true
	for i=1,256 do
		halt=_vm2:tick()
		if halt then
			add(_cmds[2],_vm2.cmd)
		end
	end
	if not halt then
		if df_eq(
			_vm2.ram[0],
			_target
			) then
			return
		end
		_iter+=1
		if _iter<10000 then
			_vm2=intcode:new(_code,_f)
			local noun=_iter\100
			local verb=_iter%100
			_vm2.ram[1]=_f(tostr(noun))
			_vm2.ram[2]=_f(tostr(verb))
			_cmds[2]={}
		end
	end
end

function _draw()
	cls()
	_vm1:draw(0,0)
	for i=1,12 do
		local idx=#_cmds[1]-i+1
		if idx<1 then
			break
		end
		print(_cmds[1][idx],0,6*i+30)
	end
	_vm2:draw(64,0)
	for i=1,12 do
		local idx=#_cmds[2]-i+1
		if idx<1 then
			break
		end
		print(_cmds[2][idx],64,6*i+30)
	end
	print(_iter,64,122)
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00077000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00700700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
