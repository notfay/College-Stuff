package model.car;

import model.engine.Engine;
import model.wheel.Wheel;

public abstract class Car {	//Blueprint = Abstract

	
	protected Engine engine;
	protected Wheel wheel;
	
	
	public Car(Engine engine, Wheel wheel) {
		super();
		this.engine = engine;
		this.wheel = wheel;
	}
	

	public Engine getEngine() {
		return engine;
	}



	public void setEngine(Engine engine) {
		this.engine = engine;
	}



	public Wheel getWheel() {
		return wheel;
	}



	public void setWheel(Wheel wheel) {
		this.wheel = wheel;
	}



	public Car() {
		// TODO Auto-generated constructor stub
	}
	
	
	
}
