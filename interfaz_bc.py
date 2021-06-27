# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:07:13 2021

@author: Derly garzon
"""
from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
from tkcalendar import Calendar,DateEntry
from datetime import datetime,timedelta
from arima_bc import *
from LSTM_bc import *
from tkinter import scrolledtext
import textwrap
'''Obtener la direccion del archivo'''
from tkinter import filedialog
from tkinter import *
import os
'''Obtener la direccion del archivo end'''

   
raiz = Tk()
raiz.title("Pronóstico de series tiempo")
raiz.config(width="400", height="600")

mi_Frame = Frame(raiz, width=400, height=450, bg='#fff')

mi_Frame.pack()
mi_Label = Label(mi_Frame, text="Pronóstico Ethereum")
mi_Label.place(x=20, y=10)
mi_Label.config(font=('Tahoma', 20,'bold'),fg='#364c55',bg='#fff')

mi_Label2 = Label(mi_Frame, text="Pronóstico usando series de tiempo")
mi_Label2.place(x=20, y=42)
mi_Label2.config(font=('Tahoma', 11),fg='#949494',bg='#fff')


mi_Label3 = Label(mi_Frame, text="Información del Algoritmo")
mi_Label3.place(x=20, y=72)
mi_Label3.config(font=('Tahoma', 11, 'bold'),fg='#364c55',bg='#fff')



#input text
'''cuadro_nombre=Entry(mi_Frame, relief='flat',highlightthickness=1)
cuadro_nombre.config(font=('Tahoma', 11),fg='#364c55', highlightbackground = "#ccc", highlightcolor= "#bbbbbb")
cuadro_nombre.insert(0, " Archivo ")
cuadro_nombre.place(x=20, y=102,width=360, height=30 )'''

'''mi_Label3 = Label(mi_Frame, text="Archivo")

mi_Label3.place(x=20, y=102)
mi_Label3.config(font=('Tahoma', 11),fg='#364c55',bg='#fff',anchor="e", justify=LEFT)'''

dirarchivo = '';
ini_date = '';

def abrir_archivo():
    '''archivo_abierto=filedialog.askopenfilename(initialdir = "/",
                title = "Seleccione archivo",filetypes = (("csv files","*.csv"),
                                                          ("jpeg files","*.jpg"),
                ("all files","*.*")))
    
    cuadro_nombre.insert(0, " Archivo - " + archivo_abierto)
    dirarchivo = archivo_abierto;'''
    coin_dataframes = {}
    
    df = pd.read_csv('C:/Users/ethereum_price.csv', parse_dates=["Date"])
    size = int(len(df) * 0.9)
    coin_dataframes['ethereum_price'] = df.sort_values('Date');
    #print(coin_dataframes['ethereum_price']['Date'])
    print('date____')
    print(coin_dataframes['ethereum_price']['Low'][len(df)-size-1])
    print(coin_dataframes['ethereum_price']['Date'][len(df)-size-1])
    date__ =  coin_dataframes['ethereum_price']['Date'][len(df)-size-1]
    '''fecha_dt = datetime.strptime(date__, '%m/%d/%Y')
    fecha_dt = fecha_dt.date()
    print(fecha_dt)'''
    cal.set_date(date__)
    ini_date = date__
    cal__.set_date(date__+ timedelta(1))
    
    mi_Label4.config(text = mi_Label4['text']+', debe ser mayor a '+str(date__) )
    print(date__.day)
    date__= date__- timedelta(1)
    date__ = str(date__.month) + '/' + str(date__.day) + '/' + str(date__.year)
    print(date__)
    #f_date__ = datetime.strptime(date__, '%m/%d/%Y')
    mi_Label_6_date.config(text =date__)
    #print('¡¡¡¡¡¡¡')
    print(mi_Label_6_date['text'])
    ''' #print(dirarchivo)'''
'''btnabrir =  Button(mi_Frame, text="Abrir archivo", command=abrir_archivo)
btnabrir.place(x=20, y=138, width=360, height=30)
btnabrir.config(font=('Tahoma', 11),fg='#fff',bg='#364c55')'''



mi_Label4 = Label(mi_Frame, text="Fecha inicial")
mi_Label4.place(x=20, y=102)
mi_Label4.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')


now_ = datetime.now().date()
year = int(now_.year)

cal = DateEntry(mi_Frame,width=30,year=year)
cal.place(x=20, y=128, width=360, height=30)
cal.config(font=('Tahoma', 11),headersbackground='#364c55', tooltipforeground='#364c55', headersforeground ='#fff', foreground='#000', background='#fff',  )


mi_Label4___1 = Label(mi_Frame, text="Fecha hasta la que desea Pronosticar")
mi_Label4___1.place(x=20, y=160)
mi_Label4___1.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')

cal__ = DateEntry(mi_Frame,width=30,year=year)
cal__.place(x=20, y=186, width=360, height=30)
cal__.config(font=('Tahoma', 11),headersbackground='#364c55' , tooltipforeground='#364c55', headersforeground ='#fff', foreground='#000', background='#fff',  )


mi_Label_6_date = Label(mi_Frame, text="")
mi_Label_6_date.place(x=20, y=585)

abrir_archivo()
    





mi_Label5 = Label(mi_Frame, text="Escoger algoritmo")
mi_Label5.place(x=20, y=220)
mi_Label5.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')

comboExample = ttk.Combobox(mi_Frame,
                            values=[
                                    "Lstm", 
                                    "Arima",
                                    ],  font=("Tahoma", 11), )

comboExample.place(x=20, y=246,width=360,height=30)
comboExample.config(font=('Tahoma', 11),foreground='#364c55')
comboExample.current(1)
mi_Frame.option_add('*TCombobox*Listbox.foreground' % comboExample, '#364c55')
mi_Frame.option_add('*TCombobox*Listbox.Font' % comboExample, 'Tahoma')




def clicked_lstm(window__):
    print(123)
    #unit = unit_entry.get()
    
    date = cal.get()
    date_fin =  cal__.get()
    date_fin_ = date_fin[-2] + date_fin[-1] 
    date_fin = date_fin[0:-2]
    date_fin = date_fin + '20' + date_fin_  
    date_ = date[-2] + date[-1] 
    date = date[0:-2]
    date = date + '20' + date_  
    print(1235)
    
    ini_date__ = mi_Label_6_date['text']
    ini_date___ = datetime.strptime(ini_date__, '%m/%d/%Y').date()
    print(12357)
    
    fecha_dt = datetime.strptime(date, '%m/%d/%Y')
    print(123578)
    fecha_dt = fecha_dt.date()
    print(123579)
    fecha_fin = datetime.strptime(date_fin, '%m/%d/%Y').date()
    print(123580)
    now = datetime.now().date()
    
    if fecha_dt > ini_date___:   
        if fecha_fin > fecha_dt :
            '''dirarchivo =cuadro_nombre.get()
            dirarchivo = dirarchivo.replace(' Archivo - ','')
            dirarchivo = dirarchivo.replace(' Archivo ','');'''
            #resulta = prediccion(int(unit),'mae',100,['Low'],'Low_time_step_' +'.png',128,.1,1,'C:/Users/ethereum_price.csv')   
            resulta = prediccion(500,'mae',100,['Low'],'Low_time_step_' +'.png',128,.1,1,'C:/Users/ethereum_price.csv')   
            print('---')
            days_ = ( fecha_fin - ini_date___).days
            days_user = ( fecha_fin - fecha_dt).days+ 1
            days_dif = days_ - days_user  ;
            print(days_)
            #resultadoarima = arimaModel(days_,dirarchivo,p,d,q,var_) 
            strResultado=''
            now__ = ini_date___
            for t in range(days_):
                print('da_____')
                print(t)
                if t >= days_dif:
                    now__ = now__ + timedelta(1)
                    print(t)
                    res = str(resulta[t])
                    print(res)
                    if t%10==0:
                         strResultado += 'Resultados\n'
                         #strResultado += str(days_dif)+'_'+str(t)+'Day ' + str(now__+timedelta(days_dif)) + ' -------- ' + res + '\n'
                    strResultado +='Day ' + str(now__+timedelta(days_dif)) + ' -------- ' + res + '\n'
            print(strResultado)
            txt2 = scrolledtext.ScrolledText(window__)
            txt2.place(x=20, y=145,width=360,height=200)
                
            txt2.config(font=('Tahoma', 10),fg='#949494',bg='#eee',padx=(10), pady=(10))
            txt2.insert(INSERT, strResultado)
            txt2.yview(END)
            txt2.config(state=DISABLED)    
        
        else:
            messagebox.showinfo(title=None, message='La fecha final Debe ser mayor a  la inicial ')
    else:
        messagebox.showinfo(title=None, message='Debe escoger una fecha inicial posterior al dia ' + str(ini_date___))

        
    return 0


def lstmIframe():
 
    x = raiz.winfo_x()
    y = raiz.winfo_y()
    print(x)
   
    newWindow = Toplevel(raiz)
    newWindow.title('Lstm')
    w = newWindow.winfo_width()
    h = newWindow.winfo_height()  
    newWindow.title("New Window")
    newWindow.geometry("%dx%d+%d+%d" % (400, 605, x+410 , y ))
    newWindow.config(bg='#fff')
    
    
    mi_Label____ = Label(newWindow, text="LSTM")
    mi_Label____.place(x=20, y=10)
    mi_Label____.config(font=('Tahoma', 20,'bold'),fg='#364c55',bg='#fff')
    
    mi_Label2__ = Label(newWindow, text="Pronóstico usando algoritmo LSTM")
    mi_Label2__.place(x=20, y=42)
    mi_Label2__.config(font=('Tahoma', 11),fg='#949494',bg='#fff')

    mi_Label3__ = Label(newWindow, text="Información para Pronosticar")
    mi_Label3__.place(x=20, y=72)
    mi_Label3__.config(font=('Tahoma', 11, 'bold'),fg='#364c55',bg='#fff')

     #timesteps   
    '''mi_Label4__ = Label(newWindow, text="No. de timesteps")
    mi_Label4__.place(x=20, y=102)
    mi_Label4__.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')
    
    cuadro_nombre__=Entry(newWindow, relief='flat',highlightthickness=1)
    cuadro_nombre__.config(font=('Tahoma', 11),fg='#364c55', highlightbackground = "#ccc", highlightcolor= "#bbbbbb")
    cuadro_nombre__.insert(0, "5")
    cuadro_nombre__.place(x=20, y=125,width=360, height=30 )
    '''
     #variable: low high open 
    '''mi_Label4__ = Label(newWindow, text="Units")
    mi_Label4__.place(x=20, y=102)
    mi_Label4__.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')
    
    comboExample__ = ttk.Combobox(newWindow,
                            values=[
                                    "500", 
                                    "130",
                                    
                                    ],  font=("Tahoma", 11), )
    comboExample__.place(x=20, y=125,width=360,height=30)
    comboExample__.config(font=('Tahoma', 11),foreground='#364c55')
    comboExample__.current(1)
    mi_Frame.option_add('*TCombobox*Listbox.foreground' % comboExample__, '#364c55')
    mi_Frame.option_add('*TCombobox*Listbox.Font' % comboExample__, 'Tahoma')
    '''

   
    #btn = Button(newWindow, text="Pronosticar", command=lambda: clicked_lstm(newWindow,comboExample__),bd=0)
    btn = Button(newWindow, text="Pronosticar", command=lambda: clicked_lstm(newWindow),bd=0)
    btn.place(x=20, y=102, width=360, height=30) 
    btn.config(font=('Tahoma', 11),fg='#fff',bg='#364c55')
    
    #prediccion(no_inputs,loss__,epoch_,feature_cols,file_,batch_,validate_,days):   

    mainloop()




def lstmIframeEv(event):
    print('lstmIframeEvalg')
    lstmIframe()

def toplevels(ventana):
    for k, v in ventana.children.items():
        if isinstance(v, Toplevel):
            print('Toplevel:', k, v)
        else:
            print('   other:', k, v)
        toplevels(v)

#boton
def clicked_arima(window__, date):

    print('ini_date:',ini_date)
    #print(window___)
    '''print('__')
    #toplevels(window___)
    #print(type(window___))
    p = int(p_entry.get())
    d = int(d_entry.get())
    q = int(q_entry.get())
    var_ = var_entry.get()'''
    #print(window___.children.entry.get())
    date = cal.get()
    date_fin =  cal__.get()
    date_fin_ = date_fin[-2] + date_fin[-1] 
    date_fin = date_fin[0:-2]
    date_fin = date_fin + '20' + date_fin_  
    date_ = date[-2] + date[-1] 
    date = date[0:-2]
    date = date + '20' + date_  
    ini_date__ = mi_Label_6_date['text']
    ini_date___ = datetime.strptime(ini_date__, '%m/%d/%Y').date()
    
    print('dddd:', date)
    fecha_dt = datetime.strptime(date, '%m/%d/%Y')
    fecha_dt = fecha_dt.date()
    fecha_fin = datetime.strptime(date_fin, '%m/%d/%Y').date()
    now = datetime.now().date()
    '''dirarchivo =cuadro_nombre.get()
    dirarchivo = dirarchivo.replace(' Archivo - ','')
    dirarchivo = dirarchivo.replace(' Archivo ','');'''
    print('archivo')
    print('--' +dirarchivo+'--')
    #print(newWindow_)
    
    #print(p)
    
    #def arimaModel(diasApredecir,pathdirectory,p,d,q,var_) 
    #arimaModel(5,'C:/Users/Derly garzon/ethereum_price.csv',5,1,0,"Low")
   
    
    
    if fecha_dt > ini_date___:   
        if fecha_fin > fecha_dt :
            print('---')
            days_ = ( fecha_fin - ini_date___).days
            days_user = ( fecha_fin - fecha_dt).days+ 1
            days_dif = days_ - days_user  ;
            print(days_)
            #resultadoarima = arimaModel(days_,dirarchivo,p,d,q,var_) 
            resultadoarima = arimaModel(days_,'C:/Users/ethereum_price.csv',1,1,1,'Low') 
            strResultado='';
            now__ = ini_date___
            for t in range(len(resultadoarima)):
                if t >= days_dif:
                    print('arr___________')
                    now__ = now__ + timedelta(1)
                    print(str(now__))
                    res = str(resultadoarima[t])
                    if t%10==0:
                         strResultado += 'Resultados\n'
                         #strResultado += str(days_dif)+'_'+str(t)+'Day ' + str(now__+timedelta(days_dif)) + ' -------- ' + res + '\n'
                    strResultado +='Day ' + str(now__+timedelta(days_dif)) + ' -------- ' + res + '\n'
            print(strResultado)
            txt = scrolledtext.ScrolledText(window__)
            txt.place(x=20, y=122,width=360,height=200)
                
            txt.config(font=('Tahoma', 10),fg='#949494',bg='#eee',padx=(10), pady=(10))
            txt.insert(INSERT, strResultado)
            txt.yview(END)
            txt.config(state=DISABLED)
        else:
             messagebox.showinfo(title=None, message='La fecha final Debe ser mayor a  la inicial ')
    else:
        messagebox.showinfo(title=None, message='Debe escoger una fecha inicial posterior al dia ' + str(ini_date___))
    #titulo.configure(text= res)
    
    

def arimaIframe():
    print(123)
    x = raiz.winfo_x()
    y = raiz.winfo_y()
    print(x)
   
    newWindow_ = Toplevel(raiz)
    w = newWindow_.winfo_width()
    h = newWindow_.winfo_height()  
    newWindow_.title("New Window")
    newWindow_.geometry("%dx%d+%d+%d" % (400, 605, x+410 , y ))
    newWindow_.config(bg='#fff')
    newWindow_.title('Arima')
    
    
    mi_Label____ = Label(newWindow_, text="ARIMA")
    mi_Label____.place(x=20, y=10)
    mi_Label____.config(font=('Tahoma', 20,'bold'),fg='#364c55',bg='#fff')
    
    mi_Label2__ = Label(newWindow_, text="Pronóstico usando algoritmo ARIMA")
    mi_Label2__.place(x=20, y=42)
    mi_Label2__.config(font=('Tahoma', 11),fg='#949494',bg='#fff')

    '''mi_Label3__ = Label(newWindow_, text="Información para Pronosticar")
    mi_Label3__.place(x=20, y=72)
    mi_Label3__.config(font=('Tahoma', 11, 'bold'),fg='#364c55',bg='#fff')

     #p, timesteps   
    mi_Label4__ = Label(newWindow_, text="Variable p - No. de timesteps")
    mi_Label4__.place(x=20, y=102)
    mi_Label4__.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')
    
    cuadro_nombre__=Entry(newWindow_, relief='flat',highlightthickness=1)
    cuadro_nombre__.config(font=('Tahoma', 11),fg='#364c55', highlightbackground = "#ccc", highlightcolor= "#bbbbbb")
    cuadro_nombre__.insert(0, "5")
    cuadro_nombre__.place(x=20, y=125,width=360, height=30 )
    
     #variable: low high open 
    mi_Label4__ = Label(newWindow_, text="Variable d - No. de veces que se integra la serie")
    mi_Label4__.place(x=20, y=160)
    mi_Label4__.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')
    
    comboExample__ = ttk.Combobox(newWindow_,
                            values=[
                                    "1", 
                                    "2",
                                    "3",
                                    ],  font=("Tahoma", 11), )

    comboExample__.place(x=20, y=183,width=360,height=30)
    comboExample__.config(font=('Tahoma', 11),foreground='#364c55')
    comboExample__.current(1)
    mi_Frame.option_add('*TCombobox*Listbox.foreground' % comboExample__, '#364c55')
    mi_Frame.option_add('*TCombobox*Listbox.Font' % comboExample__, 'Tahoma')

    

      #variable: low high open 
    mi_Label4___ = Label(newWindow_, text="Variable q - No. de medias moviles")
    mi_Label4___.place(x=20, y=218)
    mi_Label4___.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')
    
    cuadro_nombre___=Entry(newWindow_, relief='flat',highlightthickness=1)
    cuadro_nombre___.config(font=('Tahoma', 11),fg='#364c55', highlightbackground = "#ccc", highlightcolor= "#bbbbbb")
    cuadro_nombre___.insert(0, "0")
    cuadro_nombre___.place(x=20, y=241,width=360, height=30 )
    
    
     #variable: low high open 
    mi_Label4____ = Label(newWindow_, text="Variable de entrada para la predicción")
    mi_Label4____.place(x=20, y=276)
    mi_Label4____.config(font=('Tahoma', 11),fg='#364c55',bg='#fff')
    
    comboExample___ = ttk.Combobox(newWindow_,
                            values=[
                                    "Low", 
                                    "High",
                                    "Open",
                                    ],  font=("Tahoma", 11), )

    comboExample___.place(x=20, y=299,width=360,height=30)
    comboExample___.config(font=('Tahoma', 11),foreground='#364c55')
    comboExample___.current(1)
    mi_Frame.option_add('*TCombobox*Listbox.foreground' % comboExample__, '#364c55')
    mi_Frame.option_add('*TCombobox*Listbox.Font' % comboExample__, 'Tahoma')
    
    p = cuadro_nombre__.get()
    '''

    #btn_ = Button(newWindow_, text="Pronosticar", command=lambda: clicked_arima(cuadro_nombre__, comboExample__, cuadro_nombre___,comboExample___,newWindow_), bd=0)
    btn_ = Button(newWindow_, text="Pronosticar", command=lambda: clicked_arima(newWindow_, cal), bd=0)
    btn_.place(x=20, y=72, width=360, height=30) 
    btn_.config(font=('Tahoma', 11),fg='#fff',bg='#364c55')
    
    mainloop()        
  
    
def arimaIframeEv(event):
    arimaIframe()
    

  

#button.bind("<Button-1>", callback)

algoritmo__ = comboExample.get()
if algoritmo__=="Lstm":
    btn = Button(mi_Frame, text="Lstm Lstm", command=lstmIframe, bd=0)
else:
     btn = Button(mi_Frame, text="Entrenar Arima", command=arimaIframe,bd=0)



btn.place(x=20, y=290, width=360, height=30) 
btn.config(font=('Tahoma', 11),fg='#fff',bg='#364c55')


'''---'''
def changeAlgoritmo(event):
    print('changeAlgoritmo___')
    algoritmo = comboExample.get()
    textBtn = "Pronosticar " + algoritmo
    btn.config(text=textBtn)
    if algoritmo=="Lstm":
        btn.bind("<Button-1>", lstmIframeEv)
        print('LSTM')
    else:
         btn.bind("<Button-1>", arimaIframeEv)
            
        

comboExample.bind("<<ComboboxSelected>>", changeAlgoritmo)



mi_Frame_resultados = Frame(mi_Frame, width=360, height=600, bg='#eee', highlightbackground='#dedede')
mi_Frame_resultados.place(x=20, y=1006, width=360, height=200)



mi_Frame.pack()

raiz.mainloop()

