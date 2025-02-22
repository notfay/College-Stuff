package builder;

import model.engine.Engine;
import model.wheel.Wheel;

public class Director {
	
	private CarBuilder carBuilder;
	
	public Director(CarBuilder carBuilder) {
		this.carBuilder = carBuilder;
	}
	
	public Director() {
		// TODO Auto-generated constructor stub
	}
	
	public void buildSportCart() {
		this.carBuilder.setEngine(new Engine("V8", 1000, 1500));
		this.carBuilder.setWheel(new Wheel("Bridgestone", "Black", 2500));
	}
	
	public void buildFamilyCar() {
		this.carBuilder.setEngine(new Engine("Inline", 2000, 500));
		this.carBuilder.setWheel(new Wheel("GT Radial", "Red", 8500));
	}
	

}
