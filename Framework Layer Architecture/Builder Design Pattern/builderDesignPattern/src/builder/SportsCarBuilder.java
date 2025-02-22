package builder;

import model.car.SportsCar;
import model.engine.Engine;
import model.wheel.Wheel;

public class SportsCarBuilder implements CarBuilder{

	
	private SportsCar sportsCar;
	
	public SportsCarBuilder() {
		sportsCar = new SportsCar();
	}
	
	
	
	
	@Override
	public void setEngine(Engine engine) {
		sportsCar.setEngine(engine);
		
	}

	@Override
	public void setWheel(Wheel wheel) {
		sportsCar.setWheel(wheel);
		
	}
	
	
	public SportsCar getCar() {
		return sportsCar;
	}
	
	


}
