'''
Created on 4 Apr 2018

@author: Eldemin
'''
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
import pickle

class AttrUI(object):

    node = GridLayout(cols=2, row_force_default=True, row_default_height=30, width=250)
    
    def add_row(self, caption):
        label = Label(text=caption, size_hint_x=None, width=80)
        self.node.add_widget(label)
        
        text_input = TextInput()
        self.node.add_widget(text_input)
        return text_input


    def create_button(self, text, callback):
        label = Label(text='', size_hint_x=None, width=80)
        self.node.add_widget(label)
        
        btn = Button(text=text)
        btn.bind(on_press=lambda instance: callback())
        self.node.add_widget(btn)


    def __init__(self, fields, pickle_path = None, load_callback = None):
        self.node.size_hint = None, 1
        
        self.fields = []
        for field in fields:
            textInput = self.add_row(field[1])
            self.fields.append([field[0], textInput])
        
        self.create_button('reset', self.update)
        self.create_button('commit', self.commit)
        
        
        if pickle_path != None:
            self.pickle_path = pickle_path
            self.create_button('save', self.save)
            self.create_button('load', self.load)
            
            self.load_callback = load_callback

    def set_obj(self, obj):
        self.obj = obj

    def update(self):
        for field in self.fields:
            field[1].text = str(getattr(self.obj, field[0]))        
    
    
    def commit(self):
        for field in self.fields:
            setattr(self.obj, field[0], float(field[1].text))
    
    
    def save(self):
        pickle.dump(self.obj, open(self.pickle_path, "wb"))
    
    
    def load(self):
        self.obj = pickle.load(open(self.pickle_path, 'rb'))
        if self.load_callback != None:
            self.load_callback(self.obj)
                
        self.update()
    
    
    
    
    
    
    
    
    
    
    
    