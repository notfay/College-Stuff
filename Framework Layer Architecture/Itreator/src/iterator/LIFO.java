package iterator;

import java.util.ArrayList;

public class LIFO<T> implements Iterator<T> {
	
	private ArrayList<T> list;
	private int currIdx;
	
	
	public LIFO(ArrayList<T> list) {
		this.list = list;
		currIdx = list.size() - 1;
	}
	
	
	@Override
	public T getNext() {
		if (hasNext()) {
			return list.get(currIdx--);
		}
		return null;
	}

	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return currIdx >= 0;
	}

}
